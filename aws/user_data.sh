#!/bin/bash
# ---- EC2 "User data" script: paste into Advanced details -> User data ----
# Fill in the three values below. Launch N identical instances in ONE launch
# (set "Number of instances" = SHARDS); each instance picks its own shard
# from its launch index, trains, pushes results to GitHub, then shuts down.
GITHUB_REPO="zain0254/REPLACE_ME"           # e.g. zain0254/lstm-ae-adaptive-threshold
GITHUB_TOKEN="REPLACE_WITH_FINE_GRAINED_TOKEN"   # Contents: read & write, this repo only
SHARDS=4                                     # must equal "Number of instances"

exec > /var/log/atd_bootstrap.log 2>&1
for i in $(seq 1 20); do apt-get update -y && apt-get install -y git python3-venv python3-pip && break; sleep 15; done
TOKEN=$(curl -s -X PUT http://169.254.169.254/latest/api/token -H "X-aws-ec2-metadata-token-ttl-seconds: 300")
SHARD=$(curl -s -H "X-aws-ec2-metadata-token: $TOKEN" http://169.254.169.254/latest/meta-data/ami-launch-index)
cd /home/ubuntu
sudo -u ubuntu git clone "https://x-access-token:${GITHUB_TOKEN}@github.com/${GITHUB_REPO}.git" repo
cd repo
sudo -u ubuntu git config user.name "aws-runner-${SHARD}"
sudo -u ubuntu git config user.email "aws-runner@users.noreply.github.com"
sudo -u ubuntu bash aws/run_shard.sh "$SHARD" "$SHARDS"
shutdown -h now
