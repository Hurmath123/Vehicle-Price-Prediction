import numpy as np
import pandas as pd

def predict_price(data, model, encoder, feature_names, scaler=None):
    df_input = pd.DataFrame([data])
    df_input["log_mileage"] = np.log(df_input["mileage"] + 1)
    df_input["make_model"] = df_input["make"] + "_" + df_input["model"]

    # Select only required columns
    input_cat = df_input[["make", "model", "trim", "make_model"]]
    input_num = df_input[["log_mileage", "vehicle_age"]]

    # Encode
    cat_encoded = encoder.transform(input_cat)
    cat_df = pd.DataFrame(cat_encoded, columns=encoder.get_feature_names_out(), index=df_input.index)
    full_input = pd.concat([cat_df, input_num], axis=1)

    # Match training features
    for col in feature_names:
        if col not in full_input.columns:
            full_input[col] = 0
    full_input = full_input[feature_names]

    # Apply scaler if used
    if scaler:
        full_input = scaler.transform(full_input)

    log_price = model.predict(full_input)[0]
    return round(np.exp(log_price), 2)
