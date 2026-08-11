-- ============================================================
-- PostgreSQL schema — fMRI crossmodal audiovisual NLP pipeline
-- ============================================================

CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE SCHEMA IF NOT EXISTS registry;

-- Keep unqualified objects from being created outside registry.
SET search_path TO registry, public;

-- ============================================================
-- ENUM types
-- ============================================================

CREATE TYPE registry.dataset_status AS ENUM (
    'pending',
    'downloading',
    'preprocessing',
    'processed',
    'failed',
    'archived'
);

CREATE TYPE registry.modality_type AS ENUM (
    'audio',
    'visual',
    'audiovisual',
    'text'
);

CREATE TYPE registry.preproc_format AS ENUM (
    'embeddings',
    'mels',
    'mfcc',
    'text'
);

CREATE TYPE registry.proc_status AS ENUM (
    'pending',
    'running',
    'complete',
    'failed'
);

-- ============================================================
-- Datasets
-- ============================================================

CREATE TABLE registry.datasets (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    title VARCHAR(255) NOT NULL UNIQUE,
    description TEXT,
    source_url TEXT,
    doi VARCHAR(100),
    nb_of_files INTEGER NOT NULL DEFAULT 0,
    nb_of_participants INTEGER NOT NULL DEFAULT 0,
    nb_of_successful_participants INTEGER NOT NULL DEFAULT 0,
    size_gb NUMERIC(10, 2),
    fmri_tr NUMERIC(2, 2),
    languages VARCHAR(10)[],
    status registry.dataset_status NOT NULL DEFAULT 'pending',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    uploaded_at TIMESTAMPTZ,
    modified_at TIMESTAMPTZ NOT NULL DEFAULT now(),

    CONSTRAINT datasets_nb_of_files_nonnegative
        CHECK (nb_of_files >= 0),

    CONSTRAINT datasets_nb_of_participants_nonnegative
        CHECK (nb_of_participants >= 0),

    CONSTRAINT datasets_nb_of_successful_participants_nonnegative
        CHECK (nb_of_successful_participants >= 0),

    CONSTRAINT datasets_successful_participants_limit
        CHECK (nb_of_successful_participants <= nb_of_participants),

    CONSTRAINT datasets_size_gb_nonnegative
        CHECK (size_gb IS NULL OR size_gb >= 0)
);

-- ============================================================
-- Participants
-- ============================================================

CREATE TABLE registry.participants (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    dataset_id UUID NOT NULL,
    external_id VARCHAR(50),
    age INTEGER,
    sex VARCHAR(10),
    handedness VARCHAR(10),
    excluded BOOLEAN NOT NULL DEFAULT false,
    exclusion_reason TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),

    CONSTRAINT participants_dataset_fk
        FOREIGN KEY (dataset_id)
        REFERENCES registry.datasets (id)
        ON DELETE CASCADE,

    CONSTRAINT participants_dataset_external_id_unique
        UNIQUE (dataset_id, external_id),

    CONSTRAINT participants_age_nonnegative
        CHECK (age IS NULL OR age >= 0),

    CONSTRAINT participants_exclusion_reason_check
        CHECK (
            excluded = true
            OR exclusion_reason IS NULL
        )
);

CREATE INDEX participants_dataset_id_idx
    ON registry.participants (dataset_id);

-- ============================================================
-- Stimuli
-- ============================================================

CREATE TABLE registry.stimuli (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    dataset_id UUID NOT NULL,
    title VARCHAR(255) NOT NULL,
    modality registry.modality_type NOT NULL,
    language VARCHAR(10),
    duration_sec NUMERIC(10, 3),
    onset_sec NUMERIC(10, 3),
    sample_rate INTEGER,
    file_extension VARCHAR(10),
    file_size_bytes BIGINT,
    checksum VARCHAR(64),
    stored_at TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    modified_at TIMESTAMPTZ NOT NULL DEFAULT now(),

    CONSTRAINT stimuli_dataset_fk
        FOREIGN KEY (dataset_id)
        REFERENCES registry.datasets (id)
        ON DELETE CASCADE,

    CONSTRAINT stimuli_dataset_title_unique
        UNIQUE (dataset_id, title),

    CONSTRAINT stimuli_duration_nonnegative
        CHECK (duration_sec IS NULL OR duration_sec >= 0),

    CONSTRAINT stimuli_sample_rate_positive
        CHECK (sample_rate IS NULL OR sample_rate > 0),

    CONSTRAINT stimuli_file_size_nonnegative
        CHECK (file_size_bytes IS NULL OR file_size_bytes >= 0),

    CONSTRAINT stimuli_checksum_sha256_length
        CHECK (checksum IS NULL OR char_length(checksum) = 64)
);

CREATE INDEX stimuli_dataset_id_idx
    ON registry.stimuli (dataset_id);

CREATE INDEX stimuli_modality_idx
    ON registry.stimuli (modality);

-- ============================================================
-- Preprocessed stimuli
-- ============================================================

CREATE TABLE registry.preproc_stimuli (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    dataset_id UUID NOT NULL,
    stimulus_id UUID NOT NULL,
    title VARCHAR(255) NOT NULL,
    method VARCHAR(50) NOT NULL,
    pipeline_version VARCHAR(50),
    parameters JSONB,
    description VARCHAR(255),
    format registry.preproc_format,
    onset_frame INTEGER,
    dimensions INTEGER[],
    file_size_bytes BIGINT,
    checksum VARCHAR(64),
    stored_at TEXT NOT NULL,
    status registry.proc_status NOT NULL DEFAULT 'complete',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    modified_at TIMESTAMPTZ NOT NULL DEFAULT now(),

    CONSTRAINT preproc_stimuli_dataset_fk
        FOREIGN KEY (dataset_id)
        REFERENCES registry.datasets (id)
        ON DELETE CASCADE,

    CONSTRAINT preproc_stimuli_stimulus_fk
        FOREIGN KEY (stimulus_id)
        REFERENCES registry.stimuli (id)
        ON DELETE CASCADE,

    CONSTRAINT preproc_stimuli_dataset_stimulus_title_unique
        UNIQUE (dataset_id, stimulus_id, title),

    CONSTRAINT preproc_stimuli_file_size_nonnegative
        CHECK (file_size_bytes IS NULL OR file_size_bytes >= 0),

    CONSTRAINT preproc_stimuli_checksum_sha256_length
        CHECK (checksum IS NULL OR char_length(checksum) = 64),

    CONSTRAINT preproc_stimuli_dimensions_positive
        CHECK (
            dimensions IS NULL
            OR array_position(dimensions, NULL) IS NULL
               AND 0 < ALL (dimensions)
        )
);

CREATE INDEX preproc_stimuli_dataset_id_idx
    ON registry.preproc_stimuli (dataset_id);

CREATE INDEX preproc_stimuli_stimulus_id_idx
    ON registry.preproc_stimuli (stimulus_id);

CREATE INDEX preproc_stimuli_status_idx
    ON registry.preproc_stimuli (status);

CREATE INDEX preproc_stimuli_parameters_gin_idx
    ON registry.preproc_stimuli
    USING GIN (parameters);

-- ============================================================
-- Analysis reports
-- ============================================================

CREATE TABLE registry.analysis_report (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    title VARCHAR(255) NOT NULL,
    analysis_type VARCHAR(50),
    method VARCHAR(50),
    pipeline_version VARCHAR(50),
    parameters JSONB,
    description VARCHAR(255),
    dimensions INTEGER[],
    results_summary JSONB,
    file_size_bytes BIGINT,
    checksum VARCHAR(64),
    stored_at TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    modified_at TIMESTAMPTZ NOT NULL DEFAULT now(),

    CONSTRAINT analysis_report_file_size_nonnegative
        CHECK (file_size_bytes IS NULL OR file_size_bytes >= 0),

    CONSTRAINT analysis_report_checksum_sha256_length
        CHECK (checksum IS NULL OR char_length(checksum) = 64),

    CONSTRAINT analysis_report_dimensions_positive
        CHECK (
            dimensions IS NULL
            OR array_position(dimensions, NULL) IS NULL
               AND 0 < ALL (dimensions)
        )
);

CREATE INDEX analysis_report_analysis_type_idx
    ON registry.analysis_report (analysis_type);

CREATE INDEX analysis_report_parameters_gin_idx
    ON registry.analysis_report
    USING GIN (parameters);

CREATE INDEX analysis_report_results_summary_gin_idx
    ON registry.analysis_report
    USING GIN (results_summary);

-- ============================================================
-- Automatically maintain modified_at
-- ============================================================

CREATE OR REPLACE FUNCTION registry.set_modified_at()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    NEW.modified_at := now();
    RETURN NEW;
END;
$$;

CREATE TRIGGER datasets_set_modified_at
BEFORE UPDATE ON registry.datasets
FOR EACH ROW
EXECUTE FUNCTION registry.set_modified_at();

CREATE TRIGGER stimuli_set_modified_at
BEFORE UPDATE ON registry.stimuli
FOR EACH ROW
EXECUTE FUNCTION registry.set_modified_at();

CREATE TRIGGER preproc_stimuli_set_modified_at
BEFORE UPDATE ON registry.preproc_stimuli
FOR EACH ROW
EXECUTE FUNCTION registry.set_modified_at();

CREATE TRIGGER analysis_report_set_modified_at
BEFORE UPDATE ON registry.analysis_report
FOR EACH ROW
EXECUTE FUNCTION registry.set_modified_at();