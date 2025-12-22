#!/bin/bash
# Salva il branch corrente
current_branch=$(git branch --show-current)

# Fetch di tutti i cambiamenti remoti
git fetch --all --prune

# Per ogni branch remoto
for remote_branch in $(git branch -r | grep -v '\->' | sed 's/origin\///'); do
    # Se esiste già un branch locale con lo stesso nome
    if git show-ref --verify --quiet refs/heads/$remote_branch; then
        # Switch al branch locale
        git checkout $remote_branch
        # Pull dal remoto (usa --rebase se preferisci)
        git pull origin $remote_branch
    else
        # Crea nuovo branch locale che traccia il remoto
        git checkout -b $remote_branch origin/$remote_branch
    fi
done

# Torna al branch originale
git checkout $current_branch

# Ultimo fetch per sicurezza
git fetch --all