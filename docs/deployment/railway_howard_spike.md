# Railway Howard OpenContracts Spike

Project: `Howard OpenContracts Spike`

Production environment:

- Project ID: `4434ab56-9e3a-4866-96e1-b693006091ed`
- Environment ID: `3544efea-3670-48a9-b68d-12e755979085`

Services:

| Service | ID | Current role |
| --- | --- | --- |
| `opencontracts` | `16cb4bf6-566c-4909-80ba-9f9971f7e02c` | Django/ASGI web service |
| `celeryworker` | `a671a114-8983-4d32-8ba0-7e8355828304` | Background worker |
| `docling-parser` | `59324ef0-4628-4b3a-90ba-cc3e006371bb` | Parser service |
| `postgres-opencontracts` | `e3ea3dfd-3d2b-4160-b197-97c96cf4b340` | PostgreSQL |
| `redis` | `f9b45640-ac3d-4544-b8ce-41332766780a` | Redis broker/cache |
| `vector-embedder-v2` | `237b89b6-ae65-45b8-b522-609d47f10631` | New GitHub-backed embedder |
| `vector-embedder` | `ea012403-564a-4d32-9124-aceb2fe131c9` | Legacy third-party embedder, sleep mode enabled |

## Known Good State

The new vector embedder is deployed from:

- Repository: `waypoint-2/OpenContracts`
- Branch: `fix/pydantic-ai-nonstreaming-sources`
- Commit: `af0732a34c7a90fb5a0255b7c58ec110fd0448b0`
- Root directory: `compose/accelerated`
- Dockerfile: `embedder/Dockerfile`

OpenContracts and Celery are also deployed from commit
`af0732a34c7a90fb5a0255b7c58ec110fd0448b0`.

`PipelineSettings.component_settings` should point
`MicroserviceEmbedder` at:

```text
http://vector-embedder-v2.railway.internal:8000
```

## Legacy Embedder Cleanup Plan

Do not delete the legacy service until the API key dependency has been removed.
The new service currently declares `VECTOR_EMBEDDER_API_KEY`, and the legacy
service also declares `VECTOR_EMBEDDER_API_KEY`.

Safe cleanup order:

1. Move `VECTOR_EMBEDDER_API_KEY` to an independent shared variable or set it
   directly on the new embedder and OpenContracts/Celery services.
2. Verify OpenContracts, Celery, and `vector-embedder-v2` all authenticate with
   the migrated key.
3. Rename old `vector-embedder` to `vector-embedder-legacy`.
4. Rename new `vector-embedder-v2` to `vector-embedder`.
5. Update `PipelineSettings.component_settings` to the final internal hostname:
   `http://vector-embedder.railway.internal:8000`.
6. Restart OpenContracts and Celery to load the final hostname.
7. Re-run the Howard ingest readiness check and a 32-text batch embedding smoke
   test.
8. Delete the legacy service only after no services, variables, docs, or
   pipeline settings reference it.

Do not leave permanent Railway startup-command monkeypatches. The only known
startup-command monkeypatch should be confined to the sleeping legacy service
until it is safely removed.
