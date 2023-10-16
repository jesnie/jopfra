#!/bin/env bash

set -xe

git fetch origin main
git checkout main
git pull --rebase origin main
if ! git checkout -b jopfra; then
    git branch -D jopfra
    git checkout -b jopfra
fi
poetry run python -m requirements
if [[ $(git status --porcelain) ]]; then
    poetry update
    git \
        -c "user.name=Update requirements bot" \
        -c "user.email=none" \
        commit \
        -am "Update requirements."
    git push origin +jopfra
    gh pr create \
       --title "Update requirements" \
       --body "Automatic update of requirements." \
       --reviewer jesnie \
       || true
fi
