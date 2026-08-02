# Xarum background remover

Private HTTP service used by the Xarum Cloudflare Worker to remove product-photo backgrounds.

## API

- `GET /health` returns service status.
- `POST /remove` accepts raw JPEG, PNG, or WebP bytes.
- `Authorization: Bearer <SERVICE_TOKEN>` is required for `/remove`.
- Successful responses contain a transparent PNG.

The default `u2netp` model is selected for the memory limits of Render's free service.
