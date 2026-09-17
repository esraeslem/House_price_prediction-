################################################
# EV FİYAT TAHMİN MODELİ (House Price Prediction)
# Ames, Iowa - Kaggle: House Prices - Advanced Regression Techniques
# Miuul Data Scientist Bootcamp - Makine Öğrenmesi Projesi
################################################
# Görev 1: Keşifçi Veri Analizi (EDA)
# Görev 2: Feature Engineering
# Görev 3: Model Kurma
################################################

import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb

warnings.simplefilter(action="ignore")
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 220)
pd.set_option("display.float_format", lambda x: "%.3f" % x)

RANDOM_STATE = 17


################################################
# Görev 1 - Adım 1: Train ve Test veri setlerini okutup birleştiriniz.
################################################

def load_and_merge(train_path="data/train.csv", test_path="data/test.csv"):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    train["is_train"] = 1
    test["is_train"] = 0
    test["SalePrice"] = np.nan

    df = pd.concat([train, test], ignore_index=True)
    print(f"Train shape : {train.shape}")
    print(f"Test shape  : {test.shape}")
    print(f"Birleşik df : {df.shape}")
    return df


################################################
# Görev 1 - Adım 2: Numerik ve kategorik değişkenleri yakalayınız.
################################################

def grab_col_names(dataframe, cat_th=10, car_th=20):
    """
    Veri setindeki kategorik, numerik ve kategorik-fakat-kardinal
    değişkenlerin isimlerini verir. pandas 3.0'da string kolonlar
    "string" dtype ile okunabildiği için dtype == "O" yerine
    dtype.name üzerinden kontrol edilir.
    """
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtype.name in ["object", "category", "string", "str"]]
    num_but_cat = [col for col in dataframe.columns
                   if dataframe[col].nunique() < cat_th and pd.api.types.is_numeric_dtype(dataframe[col])]
    cat_but_car = [col for col in cat_cols if dataframe[col].nunique() > car_th]

    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in dataframe.columns if pd.api.types.is_numeric_dtype(dataframe[col])]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f"cat_cols: {len(cat_cols)}")
    print(f"num_cols: {len(num_cols)}")
    print(f"cat_but_car: {len(cat_but_car)}")
    print(f"num_but_cat: {len(num_but_cat)}")

    return cat_cols, num_cols, cat_but_car


################################################
# Görev 1 - Adım 3: Gerekli düzenlemeleri yapınız (tip hatası olan değişkenler gibi)
################################################

def fix_dtypes(dataframe):
    # MSSubClass, MoSold, YrSold sayısal görünen ama aslında nominal (kategorik) değişkenlerdir.
    for col in ["MSSubClass", "MoSold", "YrSold"]:
        dataframe[col] = dataframe[col].astype(str)
    return dataframe


################################################
# Görev 1 - Adım 4: Numerik ve kategorik değişkenlerin dağılımını gözlemleyiniz.
################################################

def cat_summary(dataframe, col_name):
    summary = pd.DataFrame({
        "COUNT": dataframe[col_name].value_counts(),
        "RATIO": dataframe[col_name].value_counts(normalize=True) * 100
    })
    return summary


def num_summary(dataframe, numerical_col):
    quantiles = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    return dataframe[numerical_col].describe(quantiles).to_frame().T


def plot_num_distributions(dataframe, num_cols, out_path="outputs/numeric_distributions.png"):
    cols = [c for c in num_cols if c not in ["Id", "SalePrice"]][:12]
    n = len(cols)
    ncols = 4
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 4 * nrows))
    axes = axes.flatten()
    for i, col in enumerate(cols):
        sns.histplot(dataframe[col].dropna(), kde=True, ax=axes[i], color="#4C72B0")
        axes[i].set_title(col)
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=110)
    plt.close()


################################################
# Görev 1 - Adım 5: Kategorik değişkenler ile hedef değişken incelemesi
################################################

def target_summary_with_cat(dataframe, target, categorical_col):
    return dataframe.groupby(categorical_col)[target].mean().sort_values(ascending=False).to_frame()


################################################
# Görev 1 - Adım 6: Aykırı gözlem analizi
################################################

def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    iqr = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * iqr
    low_limit = quartile1 - 1.5 * iqr
    return low_limit, up_limit


def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    return dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None)


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    # pandas 3.0'da int64 bir kolona float eşik atamak LossySetitemError fırlatır;
    # bu yüzden önce kolonu float64'e çeviriyoruz.
    if dataframe[variable].dtype.kind in "iu":
        dataframe[variable] = dataframe[variable].astype("float64")
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


################################################
# Görev 1 - Adım 7: Eksik gözlem analizi
################################################

def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=["n_miss", "ratio"])
    print(missing_df, end="\n")
    if na_name:
        return na_columns


################################################
# Görev 2 - Adım 1: Eksik ve aykırı gözlemler için gerekli işlemler
################################################

# Bu değişkenlerde NaN, "bu özellik evde yok" anlamına gelir (Ames veri seti dokümantasyonuna göre).
NONE_FILL_COLS = [
    "PoolQC", "MiscFeature", "Alley", "Fence", "FireplaceQu",
    "GarageType", "GarageFinish", "GarageQual", "GarageCond",
    "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2",
    "MasVnrType"
]

ZERO_FILL_COLS = [
    "GarageYrBlt", "GarageArea", "GarageCars",
    "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF",
    "BsmtFullBath", "BsmtHalfBath", "MasVnrArea"
]

MODE_FILL_COLS = [
    "MSZoning", "Utilities", "Exterior1st", "Exterior2nd",
    "Electrical", "KitchenQual", "Functional", "SaleType"
]


def handle_missing_values(dataframe):
    for col in NONE_FILL_COLS:
        dataframe[col] = dataframe[col].fillna("None")

    for col in ZERO_FILL_COLS:
        dataframe[col] = dataframe[col].fillna(0)

    for col in MODE_FILL_COLS:
        dataframe[col] = dataframe[col].fillna(dataframe[col].mode()[0])

    # LotFrontage: aynı mahalledeki (Neighborhood) evlerin medyanı ile doldurulur.
    dataframe["LotFrontage"] = dataframe.groupby("Neighborhood")["LotFrontage"] \
        .transform(lambda x: x.fillna(x.median()))
    dataframe["LotFrontage"] = dataframe["LotFrontage"].fillna(dataframe["LotFrontage"].median())

    return dataframe


def handle_outliers(dataframe, num_cols):
    cols = [c for c in num_cols if c not in ["Id", "SalePrice", "is_train"]]
    for col in cols:
        if check_outlier(dataframe, col):
            replace_with_thresholds(dataframe, col)
    return dataframe


################################################
# Görev 2 - Adım 2: Rare Encoder
################################################

def rare_analyser(dataframe, target, cat_cols):
    for col in cat_cols:
        print(col, ":", dataframe[col].nunique())
        print(pd.DataFrame({
            "COUNT": dataframe[col].value_counts(),
            "RATIO": dataframe[col].value_counts(normalize=True),
            "TARGET_MEAN": dataframe.groupby(col)[target].mean()
        }), end="\n\n")


def rare_encoder(dataframe, rare_perc, cat_cols):
    temp_df = dataframe.copy()
    rare_columns = [col for col in cat_cols
                     if (temp_df[col].value_counts(normalize=True) < rare_perc).sum() > 1]

    for col in rare_columns:
        freqs = temp_df[col].value_counts(normalize=True)
        rare_labels = freqs[freqs < rare_perc].index
        temp_df[col] = np.where(temp_df[col].isin(rare_labels), "Rare", temp_df[col])

    return temp_df


################################################
# Görev 2 - Adım 3: Yeni değişkenler
################################################

def add_new_features(dataframe):
    dataframe["NEW_TotalSF"] = dataframe["TotalBsmtSF"] + dataframe["1stFlrSF"] + dataframe["2ndFlrSF"]
    dataframe["NEW_HouseAge"] = dataframe["YrSold"].astype(int) - dataframe["YearBuilt"]
    dataframe["NEW_RemodAge"] = dataframe["YrSold"].astype(int) - dataframe["YearRemodAdd"]
    dataframe["NEW_GarageAge"] = dataframe["YrSold"].astype(int) - dataframe["GarageYrBlt"].replace(0, np.nan)
    dataframe["NEW_GarageAge"] = dataframe["NEW_GarageAge"].fillna(0)
    dataframe["NEW_TotalBath"] = (dataframe["FullBath"] + 0.5 * dataframe["HalfBath"] +
                                   dataframe["BsmtFullBath"] + 0.5 * dataframe["BsmtHalfBath"])
    dataframe["NEW_OverallGrade"] = dataframe["OverallQual"] * dataframe["OverallCond"]
    dataframe["NEW_TotalPorchSF"] = (dataframe["OpenPorchSF"] + dataframe["EnclosedPorch"] +
                                      dataframe["3SsnPorch"] + dataframe["ScreenPorch"] + dataframe["WoodDeckSF"])
    dataframe["NEW_HasPool"] = (dataframe["PoolArea"] > 0).astype(int)
    dataframe["NEW_HasGarage"] = (dataframe["GarageArea"] > 0).astype(int)
    dataframe["NEW_HasFireplace"] = (dataframe["Fireplaces"] > 0).astype(int)
    dataframe["NEW_Has2ndFloor"] = (dataframe["2ndFlrSF"] > 0).astype(int)
    dataframe["NEW_HasBsmt"] = (dataframe["TotalBsmtSF"] > 0).astype(int)
    dataframe["NEW_IsRemodeled"] = (dataframe["YearBuilt"] != dataframe["YearRemodAdd"]).astype(int)
    return dataframe


################################################
# Görev 2 - Adım 4: Encoding
################################################

def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe


def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    return pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)


################################################
# Görev 3: Model Kurma
################################################

def rmse_cv(model, X, y, cv=5):
    scores = cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv=cv)
    return np.sqrt(-scores)


def plot_feature_importance(model, features, out_path="outputs/feature_importance.png", top_n=20):
    importance = pd.DataFrame({"Feature": features, "Importance": model.feature_importances_})
    importance = importance.sort_values("Importance", ascending=False).head(top_n)
    plt.figure(figsize=(10, 8))
    sns.barplot(data=importance, x="Importance", y="Feature", color="#4C72B0")
    plt.title(f"LightGBM - En Önemli {top_n} Değişken")
    plt.tight_layout()
    plt.savefig(out_path, dpi=110)
    plt.close()
    return importance


################################################
# ANA AKIŞ (main)
################################################

def main():
    # ----- Görev 1: EDA -----
    df = load_and_merge()
    df = fix_dtypes(df)

    cat_cols, num_cols, cat_but_car = grab_col_names(df)
    print("\ncat_but_car:", cat_but_car)

    plot_num_distributions(df, num_cols)

    print("\n--- Görev 1 / Adım 7: Eksik Gözlemler ---")
    missing_values_table(df)

    # ----- Görev 2: Feature Engineering -----
    print("\n--- Görev 2 / Adım 1: Eksik değer ve aykırı gözlem işlemleri ---")
    df = handle_missing_values(df)
    cat_cols, num_cols, cat_but_car = grab_col_names(df)
    df = handle_outliers(df, num_cols)
    print("Kalan eksik değer sayısı:", df.drop("SalePrice", axis=1).isnull().sum().sum())

    print("\n--- Görev 2 / Adım 2: Rare Encoder ---")
    cat_cols, num_cols, cat_but_car = grab_col_names(df)
    # Rare encoding yalnızca gerçek (object/string) kategorik değişkenlere uygulanır;
    # aksi halde num_but_cat sütunları (OverallQual, FullBath vb.) stringe döner.
    object_cat_cols = [c for c in cat_cols if df[c].dtype.name in ["object", "string", "category", "str"]]
    df = rare_encoder(df, 0.01, object_cat_cols)

    print("\n--- Görev 2 / Adım 3: Yeni değişkenler ---")
    df = add_new_features(df)

    print("\n--- Görev 2 / Adım 4: Encoding ---")
    cat_cols, num_cols, cat_but_car = grab_col_names(df)
    # is_train, model akışını yönetmek için eklediğimiz yardımcı bir bayraktır; encode edilmemelidir.
    cat_cols = [col for col in cat_cols if col != "is_train"]
    binary_cols = [col for col in df.columns if df[col].dtype.name in ["object", "category", "string", "str"]
                   and df[col].nunique() == 2]
    for col in binary_cols:
        df = label_encoder(df, col)

    # cat_but_car (örn. Neighborhood) gerçek bir kimlik değişkeni olmadığından
    # (25 farklı mahalle adı, yüksek kardinalite eşiğini aştığı için buraya düşüyor)
    # bilgi kaybetmemek adına one-hot encode edilecek listeye dahil ediyoruz. "Id" hariç.
    car_cols_to_encode = [col for col in cat_but_car if col != "Id"]
    ohe_cols = [col for col in cat_cols if col not in binary_cols] + car_cols_to_encode
    df = one_hot_encoder(df, ohe_cols, drop_first=True)
    print("Encoding sonrası df boyutu:", df.shape)

    # ----- Görev 3: Model Kurma -----
    print("\n--- Görev 3 / Adım 1: Train/Test ayrımı ---")
    train_df = df[df["is_train"] == 1].drop(["is_train"], axis=1)
    test_df = df[df["is_train"] == 0].drop(["is_train", "SalePrice"], axis=1)
    print("train_df:", train_df.shape, "| test_df:", test_df.shape)

    y = train_df["SalePrice"]
    X = train_df.drop(["SalePrice", "Id"], axis=1)
    test_ids = test_df["Id"]
    X_test_final = test_df.drop(["Id"], axis=1)

    print("\n--- Görev 3 / Adım 2: Baseline model (log dönüşümsüz) ---")
    lgbm = lgb.LGBMRegressor(random_state=RANDOM_STATE, verbose=-1)
    rmse_raw = rmse_cv(lgbm, X, y, cv=5)
    print(f"5-Fold CV RMSE (log'suz): {rmse_raw.mean():,.2f}  (+/- {rmse_raw.std():,.2f})")

    print("\nBonus: Hedef değişkene log dönüşümü ---")
    y_log = np.log1p(y)
    rmse_log_scale = rmse_cv(lgbm, X, y_log, cv=5)
    print(f"5-Fold CV RMSE (log ölçeğinde): {rmse_log_scale.mean():.5f}")

    # log ölçekli tahminleri orijinal ölçeğe çevirip gerçek RMSE'yi hesaplama
    X_train, X_val, y_train_log, y_val_log = train_test_split(X, y_log, test_size=0.2, random_state=RANDOM_STATE)
    lgbm.fit(X_train, y_train_log)
    val_pred_log = lgbm.predict(X_val)
    val_pred = np.expm1(val_pred_log)
    y_val = np.expm1(y_val_log)
    rmse_log_inv = np.sqrt(mean_squared_error(y_val, val_pred))
    print(f"Log dönüşümlü model -> orijinal ölçekte validasyon RMSE: {rmse_log_inv:,.2f}")

    print("\n--- Görev 3 / Adım 3: Hiperparametre optimizasyonu ---")
    lgbm_params = {
        "learning_rate": [0.01, 0.03, 0.05, 0.1],
        "n_estimators": [300, 600, 900],
        "max_depth": [3, 5, -1],
        "num_leaves": [15, 31, 63],
        "colsample_bytree": [0.6, 0.8, 1.0],
    }
    # Arama uzayı geniş olduğundan GridSearchCV yerine RandomizedSearchCV kullanılarak
    # çalışma süresi makul seviyede tutulmuştur (sabit random_state ile tekrarlanabilir).
    lgbm_gs = RandomizedSearchCV(
        lgb.LGBMRegressor(random_state=RANDOM_STATE, verbose=-1),
        lgbm_params, n_iter=20, cv=3, n_jobs=-1, scoring="neg_mean_squared_error",
        random_state=RANDOM_STATE, verbose=0
    )
    lgbm_gs.fit(X, y_log)
    print("En iyi parametreler:", lgbm_gs.best_params_)

    final_model = lgb.LGBMRegressor(random_state=RANDOM_STATE, verbose=-1, **lgbm_gs.best_params_)
    rmse_final = rmse_cv(final_model, X, y_log, cv=5)
    print(f"Optimizasyon sonrası 5-Fold CV RMSE (log ölçeğinde): {rmse_final.mean():.5f}")

    final_model.fit(X, y_log)

    print("\n--- Görev 3 / Adım 4: Değişken önem düzeyi ---")
    importance_df = plot_feature_importance(final_model, X.columns)
    print(importance_df.head(15).to_string(index=False))

    print("\nBonus: Test verisindeki boş SalePrice değerlerinin tahmini ve submission dosyası ---")
    test_pred_log = final_model.predict(X_test_final)
    test_pred = np.expm1(test_pred_log)

    submission = pd.DataFrame({"Id": test_ids, "SalePrice": test_pred})
    submission.to_csv("outputs/submission.csv", index=False)
    print("outputs/submission.csv oluşturuldu. Boyut:", submission.shape)
    print(submission.head())

    # Özet sonuçları bir metin dosyasına da yazalım (README için)
    with open("outputs/results_summary.txt", "w", encoding="utf-8") as f:
        f.write("EV FİYAT TAHMİN MODELİ - SONUÇ ÖZETİ\n")
        f.write("=" * 45 + "\n")
        f.write(f"Baseline LightGBM 5-Fold CV RMSE (log'suz)      : {rmse_raw.mean():,.2f}\n")
        f.write(f"Log dönüşümlü model - validasyon RMSE (orijinal): {rmse_log_inv:,.2f}\n")
        f.write(f"En iyi hiperparametreler                        : {lgbm_gs.best_params_}\n")
        f.write(f"Optimizasyon sonrası 5-Fold CV RMSE (log ölçeği) : {rmse_final.mean():.5f}\n")
        f.write("\nEn önemli 15 değişken:\n")
        f.write(importance_df.head(15).to_string(index=False))

    return {
        "rmse_raw": rmse_raw.mean(),
        "rmse_log_inv": rmse_log_inv,
        "best_params": lgbm_gs.best_params_,
        "rmse_final_log": rmse_final.mean(),
    }


if __name__ == "__main__":
    results = main()
    print("\nTamamlandı:", results)
