-- Test database initialization script
-- This script is executed automatically by Docker Compose.

-- Creating a test table
CREATE TABLE IF NOT EXISTS employees (
id SERIAL PRIMARY KEY,
name VARCHAR(100) NOT NULL,
department VARCHAR(50),
salary INTEGER,
hire_date DATE
);

-- Inserting test data
INSERT INTO employees (name, department, salary, hire_date) VALUES
('Tanaka Taro', 'IT', 600000, '2020-04-01'),
('Yamada Hanako', 'HR', 550000, '2019-03-15'),
('Suzuki Ichiro', 'Finance', 700000, '2021-01-20'),
('Watanabe Yuki', 'IT', 650000, '2020-07-10'),
('Kato Akira', 'Marketing', 580000, '2022-02-01'),
('Nakamura Yui', 'IT', 620000, '2021-05-15'),
('Yoshida Saki', 'Finance', 680000, '2020-12-01'),
('Matsumoto Ryu', 'HR', 540000, '2022-08-20'),
('Inoue Kana', 'Marketing', 590000, '2021-11-10'),
('Takahashi Ken', 'IT', 710000, '2019-09-05');

-- Data verification
SELECT 'Database initialization completed. Number of employees:' as message, COUNT(*) as count FROM employees;
