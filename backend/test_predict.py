from predict import predict_price


prediction = predict_price(
    area=1500,
    bedrooms=3,
    bathrooms=2,
    stories=2,
    parking=1,
    age=5
)


print("\n========== PREDICTION ==========")
print(f"Predicted price: {prediction:.2f}")