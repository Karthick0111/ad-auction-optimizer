# Ad Auction Bid/CTR Optimization Simulator

_Real-time bidding, powered by AWS_

A Thompson Sampling bandit learns the best bid multiplier under a budget
constraint, bidding into a simulated second-price auction driven by a
LightGBM CTR model trained on real Criteo ad-click data. The "real-time"
part is genuine, not simulated-in-name-only: historical impressions are
replayed onto a Kinesis stream and scored live by a Lambda consumer, not
looped over in a plain Python script.

## What this demonstrates

- **Event-driven architecture, not batch** — Kinesis → Lambda → DynamoDB.
  This is [MarketPulse](https://github.com/Karthick0111/market-data-platform)'s
  sibling project, built deliberately differently: that one is a scheduled
  batch pipeline (Airflow/dbt), this one is a real-time event stream. Two
  different orchestration paradigms, on purpose.
- **A real ML model, not a toy one** — LightGBM trained on 200k rows of the
  actual Criteo Kaggle Display Advertising Challenge dataset, benchmarked
  against a logistic-regression baseline (hashed categoricals, the same
  trick production CTR systems use at real scale).
- **Sequential decision-making under uncertainty** — a Thompson Sampling
  multi-armed bandit picks the bid multiplier, learning online from auction
  outcomes rather than using a fixed heuristic. Tracked against the
  standard bandit regret metric (cumulative reward vs. the best fixed arm
  in hindsight).
- **Infrastructure as code, cost-consciously designed** — everything is
  Terraform. Kinesis runs in on-demand capacity mode and DynamoDB in
  on-demand billing specifically to avoid the ~$10-11/month per idle
  Kinesis shard that provisioned mode would cost between demo sessions;
  there's no SageMaker endpoint, only an on-demand Training Job. Standing
  cost between sessions is close to $0.
- **Correctness caught before it shipped, not after** — LightGBM needs a
  consistent category→integer-code mapping between training and inference;
  an early version of this pipeline silently mismatched them (a `pandas`
  dtype-mixing quirk in `.iterrows()` mangled category values on their way
  to JSON), which would have quietly degraded every prediction without
  ever raising an error. Caught by re-deriving AUC/LogLoss through the
  exact inference code path and diffing against training metrics - see
  `model/train_ctr_model.py`'s comments for the specifics.

## Architecture

```
                    Offline / training path
  Hugging Face -> data/prepare_data.py -> S3 (raw/processed)
  (Criteo_x1)                                |
                                              v
                                    Step Functions state machine
                                              |
                                              v
                                    SageMaker Training Job
                                    (LightGBM CTR model)
                                              |
                                              v
                                    S3 (model + category mappings)

                    Real-time bidding simulation path
  Streamlit -> run_trigger Lambda -> DynamoDB (init run/budget)
  (start run)         |
                       v
              producer Lambda -> Kinesis Data Stream
              (replays holdout    (on-demand, partition
               impressions)        key = run_id)
                                        |
                                        v
                             bid_consumer Lambda (Kinesis-triggered):
                             score CTR (S3 model) -> bandit picks arm
                             (DynamoDB bandit_state) -> simulate
                             competitor bids -> settle 2nd-price
                             auction -> check budget -> write result
                                        |
                                        v
                             DynamoDB (simulation_events,
                             simulation_runs, bandit_state)
                                        |
                                        v
                             Streamlit dashboard polls DynamoDB
                             (read-only IAM credentials as a secret)

  CloudWatch metrics (Kinesis/Lambda/DynamoDB) -> Grafana Cloud -> pipeline health dashboard
```

Kinesis's partition key is `run_id`, so every event for one simulation run
lands on the same shard in order - a single Lambda invocation processes
them sequentially, which is what keeps the per-run bandit state's
read-modify-write safe without needing distributed locking.

## Tech stack

Python · AWS Lambda · Kinesis · DynamoDB · S3 · Step Functions · SageMaker ·
Terraform · LightGBM · Streamlit · Grafana Cloud

## Repo structure

```
infra/                 Terraform - every AWS resource, plus the Docker-based
                        scripts that build Lambda dependency layers matching
                        Lambda's actual runtime (see infra/lambda.tf's
                        header comment for why that matters).
data/                   Pulls a Criteo CTR sample from Hugging Face (no
                        login required) - runs once, locally.
model/                  Trains the LightGBM CTR model + a logistic
                        regression baseline; outputs the model, metrics,
                        category encoding mappings, and the holdout
                        impression stream the simulation replays.
simulation/             Pure, dependency-free logic: second-price auction
                        settlement and the Thompson Sampling bandit. No AWS
                        imports - fully unit-testable, and shared by both
                        the Lambda functions and the test suite.
lambda_functions/       bid_consumer (Kinesis-triggered scorer/settler),
                        run_trigger (starts a run), producer (streams
                        holdout impressions onto Kinesis).
dashboard/              Streamlit app - CTR model performance (static,
                        reads bundled model artifacts) and live auction
                        simulation (reads DynamoDB).
monitoring/             Grafana Cloud dashboard definition (CloudWatch
                        panels for Kinesis/Lambda/DynamoDB).
tests/                  Unit tests for simulation/ - no AWS credentials
                        needed, matches what CI runs.
```

## Running it yourself

```bash
# 1. Prepare data + train the model (local, one-time)
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m data.prepare_data --n-rows 200000
python -m model.train_ctr_model

# 2. Build the Lambda dependency layer (Docker required)
./infra/build_layers.sh

# 3. Provision AWS infrastructure
cd infra
terraform init
terraform apply -target=aws_s3_bucket.artifacts   # bucket first...
cd ..
aws s3 sync model/ "s3://$(terraform -chdir=infra output -raw s3_bucket)/models/latest/" --exclude "*" --include "ctr_model.txt" --include "category_mappings.json" --include "holdout.jsonl"
./infra/package_training_code.sh "$(terraform -chdir=infra output -raw s3_bucket)"
cd infra && terraform apply                        # ...then everything else

# 4. Tear down between demo sessions (avoids any standing cost)
terraform destroy
```

Then deploy `dashboard/app.py` to Streamlit Community Cloud, pointing its
secrets at the `streamlit_readonly` IAM credentials from `terraform output`
plus the DynamoDB table names / Lambda function name from the same output.

## Known limitations / what a production version would add next

- The SageMaker training path uses the built-in scikit-learn container in
  script mode (installs `lightgbm` via `requirements.txt` at container
  start) rather than a custom container image - faster to stand up, but a
  real production pipeline would bake a purpose-built image instead.
- No API Gateway in front of `run_trigger` - the dashboard invokes it
  directly via `boto3` with IAM credentials rather than a public HTTP API.
  Fine for a single-operator demo; a real multi-user product would need
  proper request auth in front of it.
- The Grafana Cloud dashboard isn't meant to run continuously - there's
  nothing to watch between demo sessions since the whole point of the
  on-demand/no-endpoint design is near-zero standing cost. Check it
  during/after a run, not as an always-on monitor.
- `bandit_state` uses a simple read-modify-write per event rather than
  DynamoDB conditional writes - safe in practice because Kinesis's
  partition-key-per-run_id guarantees one shard (and therefore effectively
  one active writer) per run, but a genuinely multi-writer design would
  need optimistic locking.
