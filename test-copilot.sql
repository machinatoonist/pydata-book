SELECT *
FROM test_copilot
WHERE id = 1
JOIN test_copilot_2 ON test_copilot.id = test_copilot_2.id
WHERE test_copilot_2.id = 1

