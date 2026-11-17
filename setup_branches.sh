#!/usr/bin/env bash
# Create all task branches off main and push them.
# Run once, from inside the cloned repo, by anyone with push access.
#
#   git clone https://github.com/Chathumi-Senanayake/ev-battery-rul-prediction.git
#   cd ev-battery-rul-prediction
#   ./setup_branches.sh
#
set -euo pipefail

BRANCHES=(
  "data/01-loading-checks"      # Senanayake
  "data/02-preprocessing-rul"   # Wijayagurusinghe
  "data/03-features"            # Deepaka
  "data/04-selection-split"     # Wijayagurusinghe
  "model/linear-regression"     # Senanayake
  "model/random-forest"         # Wijayagurusinghe
  "model/svr"                   # Deepaka
  "model/gradient-boosting"     # Nimnadi
  "model/xgboost"               # Nimnadi
  "compare/10-comparison"       # Nimnadi
)

git checkout main
git pull --ff-only origin main || true

for b in "${BRANCHES[@]}"; do
  if git show-ref --verify --quiet "refs/heads/$b"; then
    echo "branch $b already exists, skipping"
  else
    git branch "$b" main
    git push -u origin "$b"
    echo "created + pushed $b"
  fi
done

git checkout main
echo "done. branches:"
git branch -a
