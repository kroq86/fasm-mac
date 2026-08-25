# Database

Postgres is the primary datastore. Migrations live under db/migrations.

Use `sqlx` or raw SQL with prepared statements. Connection pooling defaults to 10.
Schema changes require a numbered migration file and a rollback script.

## Migrations

Run migrations before deploy. Never edit applied migration files in place.
Track migration version in the schema_migrations table.

## Queries

Prefer indexed lookups on foreign keys. Avoid SELECT * in hot paths.
Use transactions for multi-row updates that must commit atomically.

Repeat filler for chunk overlap testing: database database database database
database database database database database database database database database
database database database database database database database database database
database database database database database database database database database
database database database database database database database database database
database database database database database database database database database
database database database database database database database database database
database database database database database database database database database
database database database database database database database database database
database database database database database database database database database
