CREATE DATABASE IF NOT EXISTS eagleeye_cnn CHARACTER SET utf8;

USE eagleeye_cnn;

CREATE TABLE IF NOT EXISTS detect_result
(
  file_id VARCHAR(32) PRIMARY KEY,
  malicious_judge BOOLEAN NOT NULL,
  malicious_chance FLOAT,
  created_at DATETIME NOT NULL,
);
