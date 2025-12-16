from app import predict_default

# Call predict_default with empty dict so app aligns features and imputes values
result = predict_default({})
print(result)
