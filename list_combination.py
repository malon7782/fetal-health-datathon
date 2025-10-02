from main import USEFUL_COLUMNS, run

results = []

for mask in range(1, min(114514, 1 << len(USEFUL_COLUMNS))):
    selected_columns = [USEFUL_COLUMNS[i] for i in range(len(USEFUL_COLUMNS)) if (mask & (1 << i)) != 0]
    # print(f"Selected columns: {selected_columns}")
    bal_acc, f1 = run(selected_columns)
    results.append((selected_columns, bal_acc, f1))

results.sort(key=lambda x: x[2], reverse=True)

for cols, bal_acc, f1 in results[:5]:
    print(f"Top 5 - Columns: {cols}, Balanced Accuracy: {bal_acc:.4f}, F1 Score: {f1:.4f}")