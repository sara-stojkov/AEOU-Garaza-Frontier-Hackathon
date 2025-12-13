CREATE TABLE GoogleQuery (
    id SERIAL PRIMARY KEY,
    user_email VARCHAR(255) NOT NULL,
    query_timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    query_text TEXT NOT NULL
);

CREATE TABLE ChatPrompt (
    id SERIAL PRIMARY KEY,
    user_email VARCHAR(255) NOT NULL,
    url TEXT,
    prompt TEXT NOT NULL,
    response TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    artifacts TEXT
);

CREATE TABLE GitCommit (
    id SERIAL PRIMARY KEY,
    user_email VARCHAR(255) NOT NULL,
    commit_hash VARCHAR(64) NOT NULL,
    committed_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);


CREATE TABLE AppUser (
    id SERIAL PRIMARY KEY,
    first_name VARCHAR(100) NOT NULL,
    last_name VARCHAR(100) NOT NULL,
    email VARCHAR(255) NOT NULL UNIQUE,
    position VARCHAR(150),
    project VARCHAR(150)
);

