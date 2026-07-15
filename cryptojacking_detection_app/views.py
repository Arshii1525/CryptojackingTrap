from django.shortcuts import render
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os
import pymysql
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from django.conf import settings
import base64
import io
import numpy as np
import os
import pandas as pd
import joblib
import random
from deap import base, creator, tools, algorithms
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import joblib


# Create your views here.
def index(request):
    return render(request,'index.html')

def admin_login(request):
    return render(request,'admin/admin_login.html')

def admin_login_action(request):
    username = request.POST.get('username')
    password = request.POST.get('password')
    if username=='Admin' and password == 'Admin':
        return render(request,'admin/admin_home.html')
    else:
        context = {'data': 'Login Failed..!'}
        return render(request,'admin/admin_login.html', context)
    
def logout(request):
    return render(request,'index.html')
    
def admin_home(request):
    return render(request,'admin/admin_home.html')

def upload_dataset(request):
    return render(request,'admin/upload_dataset.html')

global df
def upload_dataset_action(request):
    global df
    if request.method == 'POST':
        filename = request.FILES['file']
        df=pd.read_csv(filename)
        context = {'msg': 'Dataset uploaded successfully'}
        return render(request,'admin/upload_dataset.html',context)



def preprocess(request):
    raw_path = "dataset/dataset.csv"
    processed_path = "dataset/processed_dataset.csv"
    scaler_path = "model/scaler_X.joblib"

    # If processed dataset already exists, just load it
    if os.path.exists(processed_path) and os.path.exists(scaler_path):
        df = pd.read_csv(processed_path)
        scaler_X = joblib.load(scaler_path)
        msg = "Data Preprocessed successfully (Loaded)!"
    else:
        # Load raw dataset
        df = pd.read_csv(raw_path)
        df.dropna(inplace=True)

        # Ensure data types
        df['miner'] = df['miner'].astype(int)
        df['bindsnoop_PROT_TCP'] = df['bindsnoop_PROT_TCP'].astype(int)
        df['bindsnoop_PROT_UDP'] = df['bindsnoop_PROT_UDP'].astype(int)

        # Split features and labels
        X = df.drop('miner', axis=1)
        y = df['miner']

        # Scale features
        scaler_X = StandardScaler()
        X_scaled = scaler_X.fit_transform(X)

        # Save processed dataset
        X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
        X_scaled_df['miner'] = y.values
        X_scaled_df.to_csv(processed_path, index=False)

        if not os.path.exists("model"):
            os.makedirs("model")
        joblib.dump(scaler_X, scaler_path)

        msg = "Dataset Preprocessed Successfully!"

    # Train-test split (to show sizes)
    X = df.drop('miner', axis=1)
    y = df['miner']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    total_size = len(df)
    train_size = len(X_train)
    test_size = len(X_test)

    context = {
        'msg': msg,
        'records': df.head(10).values.tolist(),
        'columns': df.columns.tolist(),
        'total_size': total_size,
        'train_size': train_size,
        'test_size': test_size
    }
    return render(request, 'admin/preprocess.html', context)





def build_model(request):
    model_path = "model/cryptojacking_model.joblib"
    msg = ""

    if request.method == "POST":
        # Check if model already exists
        if os.path.exists(model_path):
            rf_model = joblib.load(model_path)
            msg = "Model Built Successfully!"
            scaler_X = joblib.load("model/scaler_X.joblib")

            # Load dataset for evaluation
            df = pd.read_csv("dataset/processed_dataset.csv")
            X = df.drop('miner', axis=1)
            y = df['miner']
            X_scaled = X.values

            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42, stratify=y
            )

        else:
            # Load preprocessed dataset
            df = pd.read_csv("dataset/processed_dataset.csv")
            X = df.drop('miner', axis=1)
            y = df['miner']

            scaler_X = StandardScaler()
            X_scaled = scaler_X.fit_transform(X)

            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42, stratify=y
            )

            # ----- GA setup -----
            def eval_rf(individual):
                n_estimators, max_depth, min_samples_split, min_samples_leaf = individual
                rf = RandomForestClassifier(
                    n_estimators=int(n_estimators),
                    max_depth=int(max_depth),
                    min_samples_split=int(min_samples_split),
                    min_samples_leaf=int(min_samples_leaf),
                    random_state=42,
                    class_weight='balanced',
                    n_jobs=-1
                )
                rf.fit(X_train, y_train)
                y_pred = rf.predict(X_test)
                acc = accuracy_score(y_test, y_pred)
                print(f"Evaluating Individual {individual} → Accuracy: {acc:.4f}")
                return acc,

            # Hyperparameter bounds
            BOUNDS = [(50, 300),  # n_estimators
                      (5, 30),    # max_depth
                      (2, 10),    # min_samples_split
                      (1, 5)]     # min_samples_leaf

            # Prevent DEAP warnings
            if not hasattr(creator, "FitnessMax"):
                creator.create("FitnessMax", base.Fitness, weights=(1.0,))
            if not hasattr(creator, "Individual"):
                creator.create("Individual", list, fitness=creator.FitnessMax)

            toolbox = base.Toolbox()
            for i, (low, up) in enumerate(BOUNDS):
                toolbox.register(f"attr_{i}", random.randint, low, up)

            toolbox.register("individual", tools.initCycle, creator.Individual,
                             [toolbox.attr_0, toolbox.attr_1, toolbox.attr_2, toolbox.attr_3], n=1)
            toolbox.register("population", tools.initRepeat, list, toolbox.individual)
            toolbox.register("evaluate", eval_rf)
            toolbox.register("mate", tools.cxTwoPoint)
            toolbox.register("mutate", tools.mutUniformInt,
                             low=[b[0] for b in BOUNDS],
                             up=[b[1] for b in BOUNDS],
                             indpb=0.2)
            toolbox.register("select", tools.selTournament, tournsize=3)

            # ----- GA execution -----
            population = toolbox.population(n=10)  # population size
            NGEN = 5  # number of generations

            print("Starting GA Optimization...")
            for gen in range(NGEN):
                print(f"\n--- Generation {gen+1}/{NGEN} ---")
                offspring = algorithms.varAnd(population, toolbox, cxpb=0.5, mutpb=0.2)
                fits = list(map(toolbox.evaluate, offspring))
                for fit, ind in zip(fits, offspring):
                    ind.fitness.values = fit
                population = toolbox.select(offspring, k=len(population))

            best_ind = tools.selBest(population, k=1)[0]
            print("\nBest Hyperparameters Found:", best_ind)

            rf_model = RandomForestClassifier(
                n_estimators=int(best_ind[0]),
                max_depth=int(best_ind[1]),
                min_samples_split=int(best_ind[2]),
                min_samples_leaf=int(best_ind[3]),
                random_state=42,
                class_weight='balanced',
                n_jobs=-1
            )
            rf_model.fit(X_train, y_train)

            if not os.path.exists("model"):
                os.makedirs("model")
            joblib.dump(rf_model, model_path)
            joblib.dump(scaler_X, "model/scaler_X.joblib")
            msg = "Model Built with GA Optimization and Saved Successfully!"

        # ----- Evaluate final model -----
        y_pred = rf_model.predict(X_test)
        accuracy = round(accuracy_score(y_test, y_pred) * 100, 2)
        class_report = classification_report(y_test, y_pred, output_dict=True)
        conf_matrix = confusion_matrix(y_test, y_pred)

        context = {
            'msg': msg,
            'accuracy': accuracy,
            'classification_report': class_report,
            'confusion_matrix': conf_matrix.tolist()
        }
        return render(request, 'admin/build_model.html', context)

    return render(request, 'admin/build_model.html')


def user_registration(request):
    return render(request,'user/user_registration.html')

def user_registration_action(request):
    username = request.POST['username']
    email = request.POST['email']
    password = request.POST['password']
    confirm_password = request.POST['confirm_password']
    if password != confirm_password:
        return render(request,'user/user_registration.html', {'msg':'Passwords do not match'})
    

    con = pymysql.connect(
        host="localhost",
        user="root",
        password="root",
        database="cryptojacking",
        charset="utf8"
    )
    cur = con.cursor()

    cur.execute("SELECT * FROM users WHERE username=%s or email=%s",(username,email))
    existing_user = cur.fetchone()

    if existing_user:
        con.close()
        return render(request,'user/user_registration.html', {'msg': 'Username or Email already exists!'})
    
    cur.execute(
        "INSERT INTO users (username, email, password) VALUES (%s, %s, %s)",
        (username, email, password)
    )
    con.commit()
    con.close()

    return render(request, 'user/user_registration.html', {'msg': 'Registration Successful!'})


def user_login(request):
    return render(request, 'user/user_login.html')

def user_login_action(request):
   
    username = request.POST['username']
    password = request.POST['password']

    con = pymysql.connect(host="localhost", user="root", password="root", database="cryptojacking", charset='utf8')
    cur = con.cursor()

    cur.execute("SELECT * FROM users WHERE username=%s AND password=%s", (username, password))
    user = cur.fetchone()
    con.close()

    if user:
        return render(request, 'user/user_home.html', {'username': username})
    else:
        return render(request, 'user/user_login.html', {'msg': 'Invalid username or password'})
    

def user_home(request):
    return render(request, 'user/user_home.html')


def analysis_graphs(request):
    """
    Generates analysis graphs for the user dashboard based on the cryptojacking detection dataset.
    Includes visual insights such as confusion matrix, CPU vs RAM usage, cache stats, BIO patterns,
    protocol usage, and miner class distribution.
    """
    msg = ""
    images = {}

    try:
        # ---------------- 📂 Load Dataset ----------------
        dataset_path = os.path.join(settings.BASE_DIR, "dataset/dataset.csv")
        processed_path = os.path.join(settings.BASE_DIR, "dataset/processed_dataset.csv")
        scaler_path = os.path.join(settings.BASE_DIR, "model/scaler_X.joblib")
        model_path = os.path.join(settings.BASE_DIR, "model/cryptojacking_model.joblib")

        # Prefer processed dataset if available
        if os.path.exists(processed_path):
            df = pd.read_csv(processed_path)
        else:
            df = pd.read_csv(dataset_path)
            df = df.dropna()

        # ---------------- 1️⃣ Confusion Matrix ----------------
        try:
            df_processed = pd.read_csv(processed_path)
            X = df_processed.drop("miner", axis=1)
            y = df_processed["miner"]

            # Load scaler and trained model
            scaler_X = joblib.load(scaler_path)
            rf_model = joblib.load(model_path)

            # Scale and predict
            X_scaled = scaler_X.transform(X)
            y_pred = rf_model.predict(X_scaled)

            # Generate confusion matrix
            cm = confusion_matrix(y, y_pred)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=rf_model.classes_)
            disp.plot(cmap="Blues", values_format="d")

            buf = io.BytesIO()
            plt.savefig(buf, format="png")
            buf.seek(0)
            images["confusion_matrix"] = base64.b64encode(buf.getvalue()).decode("utf-8")
            plt.close()
        except Exception as e:
            print("Confusion matrix plotting failed:", str(e))

        # ---------------- 2️⃣ CPU vs RAM Usage ----------------
        plt.figure(figsize=(7, 5))
        sns.scatterplot(
            x="cpuunclaimed_CPU(%)",
            y="ramusage_USED(%)",
            hue="miner",
            data=df,
            palette="coolwarm",
            alpha=0.6
        )
        plt.title("CPU vs RAM Usage")
        plt.xlabel("CPU Unclaimed (%)")
        plt.ylabel("RAM Used (%)")
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        images["cpu_ram_plot"] = base64.b64encode(buf.getvalue()).decode("utf-8")
        plt.close()

        # ---------------- 3️⃣ Cache Statistics ----------------
        plt.figure(figsize=(7, 5))
        plt.plot(df["cachestat_HITS"], label="Cache Hits", color="blue")
        plt.plot(df["cachestat_CACHED(MB)"], label="Cache Cached (MB)", color="green")
        plt.title("Cache Statistics")
        plt.xlabel("Samples")
        plt.ylabel("Values")
        plt.legend()
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        images["cache_stats"] = base64.b64encode(buf.getvalue()).decode("utf-8")
        plt.close()

        # ---------------- 4️⃣ BIO Pattern Analysis ----------------
        plt.figure(figsize=(7, 5))
        sns.scatterplot(
            x="biopattern_COUNT",
            y="biopattern_KBYTES",
            hue="miner",
            data=df,
            palette="viridis",
            alpha=0.7
        )
        plt.title("BIO Pattern Analysis")
        plt.xlabel("BIO Count")
        plt.ylabel("BIO KBytes")
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        images["bio_pattern_plot"] = base64.b64encode(buf.getvalue()).decode("utf-8")
        plt.close()

        # ---------------- 5️⃣ TCP vs UDP Protocol Usage ----------------
        plt.figure(figsize=(7, 5))
        tcp_count = df["bindsnoop_PROT_TCP"].value_counts().sum()
        udp_count = df["bindsnoop_PROT_UDP"].value_counts().sum()
        sns.barplot(x=["TCP", "UDP"], y=[tcp_count, udp_count], palette=["red", "purple"])
        plt.title("TCP vs UDP Protocol Usage")
        plt.ylabel("Count")
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        images["protocol_usage"] = base64.b64encode(buf.getvalue()).decode("utf-8")
        plt.close()

        # ---------------- 6️⃣ Miner Class Distribution ----------------
        plt.figure(figsize=(6, 4))
        sns.countplot(x="miner", data=df, palette=["orange", "green"])
        plt.title("Miner Distribution (0 = Normal, 1 = Cryptojacking)")
        plt.xlabel("Class")
        plt.ylabel("Count")
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        images["miner_distribution"] = base64.b64encode(buf.getvalue()).decode("utf-8")
        plt.close()

    except Exception as e:
        msg = f"❌ Error generating graphs: {str(e)}"

    return render(request, "user/analysis_graphs.html", {"images": images, "msg": msg})
import os, io, base64, joblib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from django.conf import settings
from django.shortcuts import render
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def enter_test_data(request):
    return render(request, 'user/enter_test_data.html')


def predict_action(request):
    if request.method == "POST":
        # Load the trained model
        rf_model = joblib.load("model/cryptojacking_model.joblib")
        
        # Load the scaler (used during training, except for 'miner')
        scaler = joblib.load("model/scaler_X.joblib")

        # Collect input values from form
        cachestat_hits = float(request.POST['cachestat_HITS'])
        cachestat_buffers = float(request.POST['cachestat_BUFFERS(MB)'])
        cachestat_cached = float(request.POST['cachestat_CACHED(MB)'])
        pidpersec = float(request.POST['pidpersec_PID/s'])
        biopattern_rnd = float(request.POST['biopattern_RND(%)'])
        biopattern_seq = float(request.POST['biopattern_SEQ(%)'])
        biopattern_count = float(request.POST['biopattern_COUNT'])
        biopattern_kbytes = float(request.POST['biopattern_KBYTES'])
        cpuunclaimed = float(request.POST['cpuunclaimed_CPU(%)'])
        ramusage = float(request.POST['ramusage_USED(%)'])
        tcpstates_new = float(request.POST['tcpstates_NEWSTATE'])
        bindsnoop_tcp = float(request.POST['bindsnoop_PROT_TCP'])
        bindsnoop_udp = float(request.POST['bindsnoop_PROT_UDP'])

        # Arrange into array
        X_input = np.array([[cachestat_hits, cachestat_buffers, cachestat_cached,
                             pidpersec, biopattern_rnd, biopattern_seq,
                             biopattern_count, biopattern_kbytes,
                             cpuunclaimed, ramusage, tcpstates_new,
                             bindsnoop_tcp, bindsnoop_udp]])

        # Scale features (except 'miner' since it's target)
        X_scaled = scaler.transform(X_input)

        # Prediction
        pred = rf_model.predict(X_scaled)[0]

        # Interpret result
        if pred == 0:
            result = "Normal Activity (System Not Cryptojacked)"
        else:
            result = "Cryptojacking Activity Detected!"

        return render(request, "user/enter_test_data.html", {"prediction": result})