# Running the training on AWS (Free plan, ~10 minutes of clicking)

The expensive step is training 324 small LSTM Autoencoders, one per (file, window size, seed). Everything after that (thresholds, metrics, statistics, figures) runs in minutes on any laptop.

The run uses **4 × m7i-flex.large** (2 vCPU, 8 GB; allowed on the AWS Free plan). Each machine trains a quarter of the jobs, pushes its results to GitHub every 15 minutes, and shuts itself down when finished.

**Cost:** about $0.10 per machine-hour × 4 machines × 3–5 hours, so a few dollars of credits.

## 0. Before you start (GitHub)

1. The repository must already be on GitHub, e.g. `zain0254/lstm-ae-adaptive-threshold`.
2. Create a **fine-grained personal access token**: GitHub → Settings → Developer settings → Personal access tokens → Fine-grained tokens → *Generate new token*.
   - Expiration: 7 days
   - Repository access: *Only select repositories* → this repo
   - Permissions → Repository permissions → **Contents: Read and write**
   - Copy the token (starts with `github_pat_`). **Delete it after the run.**

## 1. Prepare the User-data script

Open `aws/user_data.sh` from this repo and change the three lines at the top:

```bash
GITHUB_REPO="zain0254/lstm-ae-adaptive-threshold"
GITHUB_TOKEN="github_pat_..."
SHARDS=4
```

## 2. Launch the instances

AWS Console → top-right region selector: pick one (e.g. **US East (N. Virginia)**) → search **EC2** → **Launch instances**.

| Field | Value |
|---|---|
| Name | `atd-runner` |
| Application and OS Images | **Ubuntu** → *Ubuntu Server 24.04 LTS* (Free tier eligible) |
| Instance type | **m7i-flex.large** |
| Key pair | *Proceed without a key pair* |
| Network settings | defaults are fine |
| Configure storage | **20** GiB, gp3 |
| Advanced details → Shutdown behavior | **Terminate** |
| Advanced details → User data | paste the whole edited `aws/user_data.sh` |
| Summary panel (right) → **Number of instances** | **4** (must equal `SHARDS`) |

Click **Launch instance**.

> If AWS refuses 4 instances (vCPU limit), cancel, set `SHARDS=2` in the script and launch **2** instances instead. The run then takes about twice as long.

## 3. Watch progress

- On GitHub, commits named `AWS shard k/4: N score files` appear every ~15 minutes. The log of each machine is in `logs/shard_k_of_4.log`.
- When every machine has finished, the instances terminate themselves and `results/scores/` holds **324** `.npz` files.
- EC2 → Instances shows state *terminated* when a machine is done.

## 4. After the run

```bash
git pull
python scripts/evaluate.py          # tuning on validation + test metrics + statistics
python scripts/make_figures.py      # PNG figures
python scripts/make_dashboard.py    # dashboard/index.html
```

Then delete the GitHub token. Check EC2 → Instances shows nothing *running*, and EC2 → Volumes shows nothing left over.

## Troubleshooting

- **No commits after 30 min:** EC2 → select an instance → *Connect* → *EC2 Instance Connect* → `sudo tail -50 /var/log/atd_bootstrap.log` (setup) or `tail -f /home/ubuntu/repo/logs/*.log` (training).
- **A machine stopped early:** launch one instance with the same user data and `SHARDS` unchanged. Finished jobs are skipped automatically. To force a specific shard, connect and run `cd /home/ubuntu/repo && bash aws/run_shard.sh <k> 4`.
