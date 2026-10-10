#!/usr/bin/env bash
# Publish the project images to Docker Hub for linux/amd64 and linux/arm64.
#
#   docker login
#   scripts/publish-images.sh              # api + mlflow
#   scripts/publish-images.sh api          # only one image
#
# Versions come from docker-compose.yml (the tags the VM pulls); every image
# is also tagged `latest`. Each platform is built on its own and pushed by
# digest, then the digests are joined into one multi-platform tag: building
# the AutoGluon image for both platforms at once ran Docker out of memory.
# The script fails unless every tag lists every platform.
set -euo pipefail

cd "$(dirname "$0")/.."

DOCKERHUB_USER="${DOCKERHUB_USER:-juanmava}"
PLATFORMS="${PLATFORMS:-linux/amd64 linux/arm64}"
BUILDER="${BUILDER:-cf-builder}"

compose_tag() {
    sed -n "s#.*/cubierta-forestal-$1:\([^[:space:]]*\)\$#\1#p" docker-compose.yml
}

API_VERSION="$(compose_tag api)"
MLFLOW_VERSION="$(compose_tag mlflow)"
PYPROJECT_VERSION="$(sed -n 's/^version = "\(.*\)"/\1/p' services/inference-api/pyproject.toml)"

if [[ "$API_VERSION" != "$PYPROJECT_VERSION" ]]; then
    echo "docker-compose.yml uses api:$API_VERSION but pyproject.toml is $PYPROJECT_VERSION" >&2
    exit 1
fi

# push-by-digest needs a docker-container builder (not the default driver)
if ! docker buildx inspect "$BUILDER" >/dev/null 2>&1; then
    docker buildx create --name "$BUILDER" --driver docker-container
fi

# publish <name> <version> <context> [target]
publish() {
    local name="$1" version="$2" context="$3" target="${4:-}"
    local repo="docker.io/$DOCKERHUB_USER/cubierta-forestal-$name"
    local digests=() platform meta digest

    for platform in $PLATFORMS; do
        echo "==> $repo:$version · $platform"
        meta="$(mktemp)"
        docker buildx build --builder "$BUILDER" \
            --platform "$platform" \
            ${target:+--target "$target"} \
            --provenance=false \
            --output "type=image,name=$repo,push-by-digest=true,name-canonical=true,push=true" \
            --metadata-file "$meta" \
            "$context"
        digest="$(grep -o '"containerimage.digest": *"[^"]*"' "$meta" | cut -d'"' -f4)"
        rm -f "$meta"
        digests+=("$repo@$digest")
    done

    docker buildx imagetools create --builder "$BUILDER" \
        -t "$repo:$version" -t "$repo:latest" "${digests[@]}"

    local tag arch manifest
    for tag in "$version" latest; do
        manifest="$(docker buildx imagetools inspect "$repo:$tag" --format '{{json .Manifest}}')"
        for platform in $PLATFORMS; do
            arch="${platform#linux/}"
            if ! grep -Eq "\"architecture\": ?\"$arch\"" <<<"$manifest"; then
                echo "$repo:$tag is missing $platform" >&2
                exit 1
            fi
        done
        echo "OK $repo:$tag ($PLATFORMS)"
    done
}

(( $# > 0 )) || set -- api mlflow
for t in "$@"; do
    case "$t" in
        api) publish api "$API_VERSION" . api ;;
        mlflow) publish mlflow "$MLFLOW_VERSION" services/mlflow ;;
        *) echo "unknown image: $t (api | mlflow)" >&2; exit 1 ;;
    esac
done
