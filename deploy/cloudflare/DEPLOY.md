# Deploying CoRAG Cloud (Raspberry Pi 5 + Cloudflare Tunnel)

Target: `https://corag.iquantum.co`, served from the Pi via Cloudflare Tunnel.
Only `cloudflared` dials out; no ports are published on the host.

## One-time setup

1. On the Pi, create the tunnel and DNS route (writes credentials into
   `~/.cloudflared/`):

   ```bash
   cloudflared tunnel create corag
   cloudflared tunnel route dns corag corag.iquantum.co
   ```

2. Edit `deploy/cloudflare/config.yml`: replace `TUNNEL_CREDENTIALS_FILE`
   with `/etc/cloudflared/<tunnel-id>.json` (the file created in step 1).

3. Create `~/apps/corag/.env` on the Pi with real values for
   `POSTGRES_PASSWORD`, `CORAG_APP_DB_PASSWORD`, `INTERNAL_SERVICE_TOKEN`,
   `AUTH_SECRET`, `OPENAI_API_KEY` (the Pi overlay refuses to start without
   them).

## Deploy

The Pi cannot `git pull`; rsync the changed files from the workstation, then:

```bash
docker compose -f docker-compose.yml \
  -f deploy/cloudflare/docker-compose.tunnel.yml \
  -f docker-compose.pi.yml \
  --profile tunnel up -d --build
```

Sanity checks after every deploy:

- `docker compose ... config | grep -A2 'ports:'` — postgres/api/web must
  publish nothing.
- `curl -s https://corag.iquantum.co` returns the landing page.
- Register → sign in → dashboard renders the workspace.
