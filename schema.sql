CREATE DATABASE IF NOT EXISTS detect CHARACTER SET utf8;

USE detect;

CREATE TABLE IF NOT EXISTS result
(
  file_id VARCHAR(32) PRIMARY KEY,
  malicious_judge BOOLEAN NOT NULL,
  malicious_chance FLOAT,
  created_at DATETIME NOT NULL,
);
