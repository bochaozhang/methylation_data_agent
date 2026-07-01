# remove registry
sqlite3 ./registry/methyagent.db "DELETE FROM datasets; DELETE FROM task_queue;"

# remove downloaded sets
rm -r -I ./data/*

