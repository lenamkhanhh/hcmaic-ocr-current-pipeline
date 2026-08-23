# Security and publishing boundary

This handoff is designed to be shareable, but it is not a data release.

- Do not add `.env`, API keys, Kaggle tokens, Elasticsearch credentials,
  signed URLs, private model files, snapshots, keyframes, or generated output.
- Configure Elasticsearch credentials through environment-variable names only.
  URLs must not contain userinfo, query strings, fragments, or secrets.
- Replace the notebook placeholders only in a private local copy or a private
  execution workspace.
- Keep the repository private until a human has reviewed the staged diff and
  the remote visibility. Public publication is a separate approval from
  creating this local handoff.
- If a secret was ever committed, rotate it; deleting the line is not enough.

