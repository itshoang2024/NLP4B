create table if not exists public.video_processing_progress (
    video_id text primary key,
    runner text not null,
    keyframe_extraction boolean not null default false,
    keyframe_upload boolean not null default false,
    embedding boolean not null default false,
    object_detection boolean not null default false,
    ocr boolean not null default false,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),

    constraint video_processing_progress_video_id_chk
        check (video_id ~ '^[A-Za-z0-9_-]{6,20}$'),

    constraint video_processing_progress_runner_chk
        check (runner in ('hoang', 'nanh', 'lam', 'binh'))
);

create or replace function public.set_updated_at()
returns trigger
language plpgsql
as $$
begin
    new.updated_at = now();
    return new;
end;
$$;

drop trigger if exists trg_video_processing_progress_updated_at
on public.video_processing_progress;

create trigger trg_video_processing_progress_updated_at
before update on public.video_processing_progress
for each row
execute function public.set_updated_at();

create index if not exists idx_video_processing_progress_runner
on public.video_processing_progress (runner);