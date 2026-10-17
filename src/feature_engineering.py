def add_features(df):
    """
    Create additional features for the House Prices dataset.
    The same transformations should be applied to train and test.
    """

    df = df.copy()

    # Total house area
    df["TotalSF"] = (
        df["TotalBsmtSF"].fillna(0)
        + df["1stFlrSF"].fillna(0)
        + df["2ndFlrSF"].fillna(0)
    )

    # Total bathrooms, counting half bathrooms as 0.5
    df["TotalBathrooms"] = (
        df["FullBath"].fillna(0)
        + 0.5 * df["HalfBath"].fillna(0)
        + df["BsmtFullBath"].fillna(0)
        + 0.5 * df["BsmtHalfBath"].fillna(0)
    )

    # Total porch and deck area
    porch_columns = [
        "OpenPorchSF",
        "3SsnPorch",
        "EnclosedPorch",
        "ScreenPorch",
        "WoodDeckSF"
    ]

    df["TotalPorchSF"] = (
        df[porch_columns].fillna(0).sum(axis=1)
    )

    # House age when sold
    df["HouseAge"] = (
        df["YrSold"] - df["YearBuilt"]
    )

    # Years since remodeling
    df["RemodAge"] = (
        df["YrSold"] - df["YearRemodAdd"]
    )

    # Interaction between house quality and living area
    df["OverallQual_SF"] = (
        df["OverallQual"].fillna(0)
        * df["GrLivArea"].fillna(0)
    )

    return df
