-- Drop tables if they exist (optional, for clean creation)
DROP TABLE IF EXISTS song_ratings CASCADE;
DROP TABLE IF EXISTS rl_weights CASCADE;
DROP TABLE IF EXISTS rl_q_table CASCADE;
DROP TABLE IF EXISTS users CASCADE;

-- Create the users table
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    username VARCHAR NOT NULL UNIQUE,
    email VARCHAR NOT NULL UNIQUE,
    desired_mood VARCHAR NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    favourite_music_genres JSONB NOT NULL DEFAULT '[]'
);

-- Create RLQTable
CREATE TABLE rl_q_table (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL,
    song_id VARCHAR NOT NULL,
    mood VARCHAR NOT NULL,
    prev_rating INTEGER NOT NULL,
    arousal FLOAT,
    valence FLOAT,
    weight_similar_users_music_prefs_idx INTEGER NOT NULL,
    weight_current_user_mood_idx INTEGER NOT NULL,
    weight_desired_mood_after_listening_idx INTEGER NOT NULL,
    q_value FLOAT DEFAULT 0.0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_user FOREIGN KEY (user_id) REFERENCES users(id),
    CONSTRAINT check_arousal_range CHECK (arousal >= -1 AND arousal <= 1),
    CONSTRAINT check_valence_range CHECK (valence >= -1 AND valence <= 1),
    CONSTRAINT check_similar_users_idx CHECK (weight_similar_users_music_prefs_idx >= 0 AND weight_similar_users_music_prefs_idx <= 4),
    CONSTRAINT check_current_mood_idx CHECK (weight_current_user_mood_idx >= 0 AND weight_current_user_mood_idx <= 4),
    CONSTRAINT check_desired_mood_idx CHECK (weight_desired_mood_after_listening_idx >= 0 AND weight_desired_mood_after_listening_idx <= 4),
    CONSTRAINT check_prev_rating CHECK (prev_rating >= 1 AND prev_rating <= 5)
);
CREATE INDEX idx_rl_q_table_song_id ON rl_q_table(song_id);

-- Recreate rl_weights with SERIAL PRIMARY KEY
CREATE TABLE rl_weights (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL,
    mood VARCHAR NOT NULL,
    weight_similar_users_music_prefs FLOAT NOT NULL,
    weight_current_user_mood FLOAT NOT NULL,
    weight_desired_mood_after_listening FLOAT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_user FOREIGN KEY (user_id) REFERENCES users(id),
    CONSTRAINT uq_rl_weights_user_id_mood UNIQUE (user_id, mood),
    CONSTRAINT check_similar_users_non_negative CHECK (weight_similar_users_music_prefs >= 0),
    CONSTRAINT check_current_mood_non_negative CHECK (weight_current_user_mood >= 0),
    CONSTRAINT check_desired_mood_non_negative CHECK (weight_desired_mood_after_listening >= 0)
);

-- Create SongRating
CREATE TABLE song_ratings (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL,
    song_id VARCHAR NOT NULL,
    rating INTEGER NOT NULL,
    mood_at_rating VARCHAR NOT NULL,
    arousal FLOAT CHECK (arousal >= -1 AND arousal <= 1),
    valence FLOAT CHECK (valence >= -1 AND valence <= 1),
    context VARCHAR,
    danceability FLOAT,
    energy FLOAT,
    acousticness FLOAT,
    instrumentalness FLOAT,
    speechiness FLOAT,
    liveness FLOAT,
    tempo FLOAT,
	loudness FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_user FOREIGN KEY (user_id) REFERENCES users(id),
    CONSTRAINT check_rating CHECK (rating >= 1 AND rating <= 5)
);

-- Create an index on updated_at for rl_q_table (optional, for performance)
CREATE INDEX rl_q_table_updated_at_idx ON rl_q_table (updated_at);

-- Recreate index
CREATE INDEX rl_weights_updated_at_idx ON rl_weights (updated_at);

INSERT INTO users (username, email, desired_mood, favourite_music_genres)
VALUES ('vishwa98', 'vishwa@gmail.com', 'Calm', '["Pop", "Classical"]');

SELECT * FROM users
SELECT * FROM rl_q_table
SELECT * FROM rl_weights
SELECT * FROM song_ratings