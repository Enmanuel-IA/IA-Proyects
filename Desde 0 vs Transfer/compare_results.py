import json

with open("scratch_results.json", "r", encoding="utf-8") as f:
    scratch = json.load(f)

with open("transfer_results.json", "r", encoding="utf-8") as f:
    transfer = json.load(f)

print("\nCOMPARACION DE MODELOS\n")

print(f"Modelo desde cero:")
print(f"  Test Accuracy: {scratch['test_accuracy']:.4f}")
print(f"  Test Loss: {scratch['test_loss']:.4f}")
print(f"  Train Accuracy final: {scratch['final_train_accuracy']:.4f}")
print(f"  Val Accuracy final: {scratch['final_val_accuracy']:.4f}")

print()

print(f"Modelo con transfer learning:")
print(f"  Test Accuracy: {transfer['test_accuracy']:.4f}")
print(f"  Test Loss: {transfer['test_loss']:.4f}")
print(f"  Train Accuracy final: {transfer['final_train_accuracy']:.4f}")
print(f"  Val Accuracy final: {transfer['final_val_accuracy']:.4f}")

print("\nMEJOR MODELO:")
if transfer["test_accuracy"] > scratch["test_accuracy"]:
    print("  Transfer Learning obtuvo mejor accuracy.")
elif transfer["test_accuracy"] < scratch["test_accuracy"]:
    print("  El modelo entrenado desde cero obtuvo mejor accuracy.")
else:
    print("  Ambos modelos obtuvieron el mismo accuracy.")