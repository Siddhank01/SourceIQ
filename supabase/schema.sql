create table if not exists public.sessions (
  id text primary key,
  owner text not null,
  title text not null,
  description text not null default '',
  saved boolean not null default false,
  messages jsonb not null default '[]'::jsonb,
  sources jsonb not null default '[]'::jsonb,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create index if not exists sessions_owner_updated_idx on public.sessions (owner, updated_at desc);

create table if not exists public.document_chunks (
  session_id text not null,
  chunk_id text not null,
  content text not null,
  metadata jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now(),
  primary key (session_id, chunk_id)
);

create index if not exists document_chunks_session_idx on public.document_chunks (session_id);

alter table public.sessions enable row level security;
alter table public.document_chunks enable row level security;

-- The Python API uses SUPABASE_SERVICE_ROLE_KEY and performs server-side access.
-- Do not expose that key to the frontend.
