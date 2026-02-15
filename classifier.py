"""
Sport vs Politics Text Classifier
===================================
A comprehensive comparison of Naive Bayes, Logistic Regression, and SVM
using Bag of Words, TF-IDF, and N-Gram feature representations.

Author: ML Research Team
Date: February 2026
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score, precision_score, recall_score,
    roc_curve, auc
)
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')
import json, os

# ─────────────────────────────────────────────
# 1. DATASET CREATION (BBC-style synthetic data)
# ─────────────────────────────────────────────
SPORTS_TEXTS = [
    "The football team secured a dramatic last-minute victory in the championship final with a stunning header from the striker.",
    "Manchester United lost their Premier League match against Chelsea, dropping three crucial points in the title race.",
    "LeBron James scored 35 points to lead the Lakers to a playoff win, dominating the fourth quarter.",
    "The tennis tournament at Wimbledon saw an upset as the world number one was knocked out in the second round.",
    "Serena Williams announced her retirement from professional tennis after a remarkable 23-year career.",
    "The Olympic swimming relay team set a new world record, breaking the previous mark by nearly a second.",
    "Roger Federer's backhand technique is studied by aspiring tennis professionals worldwide.",
    "Arsenal's manager fielded a defensive lineup but still managed to claim three points from the derby.",
    "The Tour de France leader crashed on the descent and was airlifted to hospital with minor injuries.",
    "The cricket team won the Ashes series after a commanding performance in the final test match.",
    "NBA playoffs: the Warriors defeated the Celtics 112-108 in a thrilling game seven.",
    "The Formula 1 driver set the fastest lap in qualifying and will start from pole position.",
    "The boxer retained his world heavyweight championship title with a knockout in the eighth round.",
    "The ski jump athlete achieved a record distance at the Winter Olympics in Beijing.",
    "The marathon runner finished in two hours and four minutes, claiming a national record.",
    "The rugby team's fly-half kicked four penalties to secure a narrow win over their rivals.",
    "The baseball pitcher threw a no-hitter, striking out twelve batters in a dominant performance.",
    "Champions League semi-final: Real Madrid overcame Bayern Munich with a 3-2 aggregate win.",
    "The golf champion carded a final-round 65 to win the Masters tournament by two strokes.",
    "The swimmer broke the 100m freestyle world record at the World Aquatics Championships.",
    "The basketball team's coach was dismissed following a ten-game losing streak.",
    "The volleyball national team won gold at the Pan American Games in a five-set thriller.",
    "The young footballer signed a five-year contract extension worth 300,000 pounds per week.",
    "The athletics federation suspended the sprinter following a positive doping test.",
    "The horse racing favourite won the Grand National in record time after a flawless run.",
    "The American football quarterback threw four touchdowns in a commanding Super Bowl victory.",
    "The ice hockey goalie made 45 saves to earn a shutout in the Stanley Cup playoff match.",
    "The gymnast earned a perfect 10 from the judges for her floor routine at the World Championships.",
    "The rowing eight won gold at the World Rowing Championships, their third consecutive title.",
    "The cycling sprinter claimed the stage win in the Giro d'Italia with a blistering final sprint.",
    "The goalkeeper saved a penalty in extra time to send his side into the cup final.",
    "The cricketer was caught behind for a golden duck, prompting boos from the home crowd.",
    "The decathlon champion scored over 9000 points across all ten events at the European Games.",
    "The weightlifter lifted a combined total of 440kg to claim the Olympic gold medal.",
    "The snooker player compiled a maximum 147 break in the first round of the World Championship.",
    "The football referee was criticised for failing to award a clear penalty kick in the final minutes.",
    "The alpine skier crashed at 130km/h but miraculously walked away without injury.",
    "The women's World Cup final drew over 1.2 billion television viewers worldwide.",
    "The table tennis champion extended his unbeaten run to 47 matches with another dominant display.",
    "The badminton player's overhead smash has been clocked at over 400km/h, the world's fastest.",
    "The triathlete finished the gruelling Ironman race despite suffering from cramp in the swim leg.",
    "Transfer window: the Premier League club spent over 150 million pounds on new signings.",
    "The archery team won silver at the Asian Games, missing gold by a single point.",
    "The disabled sprinter broke the Paralympic world record in the 100m T44 category.",
    "The American athlete won the long jump with a leap of 8.95 metres at the World Championships.",
    "The surfing champion rode a 20-metre wave to claim the Big Wave Award for 2025.",
    "The baseball team's general manager was fired after four consecutive seasons outside the playoffs.",
    "The darts player hit nine consecutive 180s to win the World Darts Championship in dramatic fashion.",
    "The stadium renovation project will increase capacity to 90,000 fans for next season.",
    "The youth academy graduate made his first-team debut, scoring twice against a top-four rival.",
]

POLITICS_TEXTS = [
    "The prime minister announced sweeping tax reforms aimed at reducing the national deficit.",
    "Parliament voted to approve the new immigration bill after weeks of heated debate in the Commons.",
    "The senator called for an independent inquiry into alleged corruption within the defence ministry.",
    "The president signed an executive order extending environmental protections across federal lands.",
    "Tensions between the two nations escalated after diplomatic talks broke down over trade tariffs.",
    "The opposition leader accused the government of mismanaging the national health service budget.",
    "The UN Security Council passed a resolution demanding an immediate ceasefire in the conflict zone.",
    "The chancellor delivered the autumn budget, raising income tax thresholds for low earners.",
    "The foreign minister met her counterpart in Brussels to finalise the bilateral trade agreement.",
    "The election commission confirmed voter turnout exceeded 68 percent in the general election.",
    "The prime minister survived a vote of no confidence by a margin of just five votes.",
    "The Supreme Court ruled that the government's surveillance programme violated constitutional rights.",
    "The governor declared a state of emergency following widespread protests in the capital city.",
    "NATO allies agreed to increase defence spending to three percent of GDP by 2030.",
    "The senator was indicted on charges of bribery and money laundering related to lobbying activities.",
    "The peace agreement was signed by both factions, ending a three-year civil conflict.",
    "The central bank raised interest rates by 50 basis points to combat rising inflation.",
    "The government proposed a new green energy policy to reach net zero emissions by 2040.",
    "The finance committee published a report criticising wasteful public spending in the military.",
    "The president-elect named her cabinet, with several women appointed to senior ministerial roles.",
    "The political party's leadership election was won by the left-wing candidate by a narrow margin.",
    "The referendum on independence attracted over 80 percent voter turnout, a record for the region.",
    "Sanctions were imposed on the country following evidence of human rights violations.",
    "The whistleblower leaked classified documents exposing a wide-scale government surveillance network.",
    "The trade war between the world's two largest economies escalated with new tariff announcements.",
    "The mayor announced plans to tackle homelessness with a 500-million-pound housing initiative.",
    "The ambassador was expelled following allegations of espionage against the host government.",
    "The parliamentary committee launched an inquiry into the lobbying practices of pharmaceutical firms.",
    "The party conference saw members debate climate policy, immigration reform and the NHS.",
    "The opposition proposed doubling the minimum wage to address growing income inequality.",
    "The president signed a landmark bipartisan infrastructure bill worth two trillion dollars.",
    "The regional government declared autonomy in defiance of the central government's authority.",
    "The justice minister proposed sentencing reforms to reduce prison overcrowding.",
    "The head of state visited the flood-affected regions and pledged emergency relief funding.",
    "The new law banning single-use plastics came into effect, drawing praise from environmental groups.",
    "The constitutional amendment was ratified by a two-thirds majority in the national assembly.",
    "The prime minister reshuffled the cabinet, bringing in new faces for health and education.",
    "The ruling coalition collapsed after a key party withdrew support over the budget disagreement.",
    "The youth vote surged in the election, with under-30s backing the progressive candidate by 70 percent.",
    "Transparency International ranked the country 12th in its annual corruption perceptions index.",
    "The government introduced legislation to cap energy prices for vulnerable households.",
    "The international court issued an arrest warrant for the former president on war crimes charges.",
    "The senator's filibuster lasted 14 hours, delaying the vote on the controversial defence budget.",
    "The political crisis deepened as three ministers resigned within 48 hours over policy disagreements.",
    "The government launched a major consultation on pension reform, affecting millions of retirees.",
    "The foreign policy speech outlined a new doctrine prioritising multilateral diplomatic engagement.",
    "The data protection commissioner fined the ministry for breaching citizens' privacy rights.",
    "The new electoral boundary changes were criticised as gerrymandering by the opposition parties.",
    "The global summit on food security ended with a joint communiqué pledging aid to developing nations.",
    "The whistleblower was granted asylum after fleeing allegations of treason in his home country.",
]

def build_dataset():
    texts = SPORTS_TEXTS + POLITICS_TEXTS
    labels = ["Sports"] * len(SPORTS_TEXTS) + ["Politics"] * len(POLITICS_TEXTS)
    df = pd.DataFrame({"text": texts, "label": labels})
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    return df

# ─────────────────────────────────────────────
# 2. FEATURE VECTORISERS
# ─────────────────────────────────────────────
def get_vectorizers():
    return {
        "Bag of Words": CountVectorizer(stop_words='english', max_features=3000),
        "TF-IDF Unigram": TfidfVectorizer(stop_words='english', max_features=3000, ngram_range=(1,1)),
        "TF-IDF Bigram": TfidfVectorizer(stop_words='english', max_features=3000, ngram_range=(1,2)),
    }

# ─────────────────────────────────────────────
# 3. CLASSIFIERS
# ─────────────────────────────────────────────
def get_classifiers():
    return {
        "Naive Bayes": MultinomialNB(alpha=0.5),
        "Logistic Regression": LogisticRegression(max_iter=500, C=1.0, random_state=42),
        "Linear SVM": LinearSVC(C=1.0, max_iter=2000, random_state=42),
    }

# ─────────────────────────────────────────────
# 4. EVALUATION HELPERS
# ─────────────────────────────────────────────
def evaluate(clf, vec, X_train, X_test, y_train, y_test):
    X_tr = vec.fit_transform(X_train)
    X_te = vec.transform(X_test)
    clf.fit(X_tr, y_train)
    y_pred = clf.predict(X_te)

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, pos_label="Sports")
    rec  = recall_score(y_test, y_pred, pos_label="Sports")
    f1   = f1_score(y_test, y_pred, pos_label="Sports")
    cm   = confusion_matrix(y_test, y_pred, labels=["Sports","Politics"])
    return acc, prec, rec, f1, cm, y_pred

def cross_val(clf, vec, X, y, cv=5):
    pipe = Pipeline([("vec", vec), ("clf", clf)])
    skf  = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    scores = cross_val_score(pipe, X, y, cv=skf, scoring="accuracy")
    return scores.mean(), scores.std()

# ─────────────────────────────────────────────
# 5. PLOTTING UTILITIES
# ─────────────────────────────────────────────
def plot_confusion_matrices(results, outdir):
    fig, axes = plt.subplots(3, 3, figsize=(15, 13))
    fig.suptitle("Confusion Matrices – All Model / Feature Combinations", fontsize=16, fontweight='bold', y=1.01)
    classes = ["Sports", "Politics"]
    cmap = sns.color_palette("Blues", as_cmap=True)

    vecs = list(results.keys())
    clfs = list(results[vecs[0]].keys())

    for vi, vec_name in enumerate(vecs):
        for ci, clf_name in enumerate(clfs):
            ax = axes[ci][vi]
            cm = results[vec_name][clf_name]["cm"]
            sns.heatmap(cm, annot=True, fmt="d", cmap=cmap,
                        xticklabels=classes, yticklabels=classes,
                        ax=ax, cbar=False, linewidths=.5, linecolor='gray')
            ax.set_title(f"{clf_name}\n({vec_name})", fontsize=9, fontweight='bold')
            ax.set_xlabel("Predicted", fontsize=8)
            ax.set_ylabel("Actual", fontsize=8)
            ax.tick_params(labelsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "confusion_matrices.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("  [✓] confusion_matrices.png saved")

def plot_metrics_comparison(results, outdir):
    rows = []
    for vec_name, clf_dict in results.items():
        for clf_name, metrics in clf_dict.items():
            rows.append({
                "Feature": vec_name,
                "Classifier": clf_name,
                "Accuracy": metrics["acc"],
                "Precision": metrics["prec"],
                "Recall": metrics["rec"],
                "F1": metrics["f1"],
            })
    df_r = pd.DataFrame(rows)

    metrics_list = ["Accuracy", "Precision", "Recall", "F1"]
    colors = ["#2196F3", "#FF5722", "#4CAF50"]
    clf_names = df_r["Classifier"].unique()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Classifier Performance Across Feature Representations", fontsize=15, fontweight='bold')

    for idx, metric in enumerate(metrics_list):
        ax = axes[idx // 2][idx % 2]
        vec_names = df_r["Feature"].unique()
        x = np.arange(len(vec_names))
        width = 0.25
        for i, (clf_name, color) in enumerate(zip(clf_names, colors)):
            vals = [df_r[(df_r["Feature"]==v) & (df_r["Classifier"]==clf_name)][metric].values[0]
                    for v in vec_names]
            bars = ax.bar(x + i*width, vals, width, label=clf_name, color=color, alpha=0.85, edgecolor='white')
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                        f"{val:.2f}", ha='center', va='bottom', fontsize=7.5)
        ax.set_title(metric, fontsize=12, fontweight='bold')
        ax.set_xticks(x + width)
        ax.set_xticklabels([v.replace(" ", "\n") for v in vec_names], fontsize=9)
        ax.set_ylim(0.5, 1.05)
        ax.set_ylabel("Score")
        ax.legend(fontsize=8)
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "metrics_comparison.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("  [✓] metrics_comparison.png saved")

def plot_cross_val(cv_results, outdir):
    rows = []
    for vec_name, clf_dict in cv_results.items():
        for clf_name, (mean, std) in clf_dict.items():
            rows.append({"Feature+Classifier": f"{clf_name}\n({vec_name})", "Mean": mean, "Std": std})
    df_cv = pd.DataFrame(rows).sort_values("Mean", ascending=True)

    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(df_cv)))
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.barh(df_cv["Feature+Classifier"], df_cv["Mean"],
                   xerr=df_cv["Std"], color=colors, edgecolor='white',
                   alpha=0.9, capsize=4)
    for bar, (_, row) in zip(bars, df_cv.iterrows()):
        ax.text(row["Mean"] + 0.005, bar.get_y() + bar.get_height()/2,
                f"{row['Mean']:.3f} ± {row['Std']:.3f}",
                va='center', fontsize=8)
    ax.set_xlim(0.5, 1.08)
    ax.set_xlabel("5-Fold CV Accuracy", fontsize=12)
    ax.set_title("Cross-Validation Accuracy by Model & Feature Combination", fontsize=13, fontweight='bold')
    ax.axvline(x=0.9, color='red', linestyle='--', alpha=0.4, label="0.90 threshold")
    ax.legend(fontsize=9)
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "cross_val_accuracy.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("  [✓] cross_val_accuracy.png saved")

def plot_dataset_dist(df, outdir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Label distribution
    label_counts = df["label"].value_counts()
    axes[0].pie(label_counts, labels=label_counts.index,
                autopct='%1.1f%%', colors=["#2196F3","#FF5722"],
                startangle=90, wedgeprops={'edgecolor':'white','linewidth':2})
    axes[0].set_title("Class Distribution", fontsize=13, fontweight='bold')

    # Text length distribution
    df["word_count"] = df["text"].apply(lambda t: len(t.split()))
    for label, color in zip(["Sports","Politics"],["#2196F3","#FF5722"]):
        subset = df[df["label"]==label]["word_count"]
        axes[1].hist(subset, bins=15, alpha=0.6, label=label, color=color, edgecolor='white')
    axes[1].set_xlabel("Word Count per Document", fontsize=11)
    axes[1].set_ylabel("Frequency", fontsize=11)
    axes[1].set_title("Word Count Distribution by Class", fontsize=13, fontweight='bold')
    axes[1].legend()
    axes[1].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "dataset_analysis.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("  [✓] dataset_analysis.png saved")

def plot_top_features(vectorizers_fit, outdir, top_n=15):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f"Top {top_n} Discriminative Features per Representation", fontsize=14, fontweight='bold')

    for ax, (vec_name, (vec, X_tr, y_tr)) in zip(axes, vectorizers_fit.items()):
        from sklearn.feature_selection import chi2
        chi2_vals, p_vals = chi2(X_tr, y_tr)
        feat_names = np.array(vec.get_feature_names_out())
        top_idx = np.argsort(chi2_vals)[-top_n:][::-1]
        top_feats = feat_names[top_idx]
        top_chi2  = chi2_vals[top_idx]

        colors = plt.cm.Blues(np.linspace(0.4, 0.9, top_n))[::-1]
        ax.barh(range(top_n), top_chi2[::-1], color=colors[::-1])
        ax.set_yticks(range(top_n))
        ax.set_yticklabels(top_feats[::-1], fontsize=8)
        ax.set_title(vec_name, fontsize=11, fontweight='bold')
        ax.set_xlabel("Chi² Score", fontsize=9)
        ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "top_features.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("  [✓] top_features.png saved")

# ─────────────────────────────────────────────
# 6. MAIN PIPELINE
# ─────────────────────────────────────────────
def main():
    outdir = "/home/claude/sport_politics_classifier/outputs"
    os.makedirs(outdir, exist_ok=True)

    print("=" * 60)
    print("  Sport vs Politics Text Classifier")
    print("=" * 60)

    # Build dataset
    df = build_dataset()
    print(f"\n[Dataset] Total samples: {len(df)}")
    print(df["label"].value_counts().to_string())

    X = df["text"].values
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )
    print(f"\n  Training: {len(X_train)} | Test: {len(X_test)}")

    vectorizers = get_vectorizers()
    classifiers = get_classifiers()

    results    = {}
    cv_results = {}
    vf_data    = {}  # for feature importance plots

    print("\n[Training & Evaluation]")
    for vec_name, vec in vectorizers.items():
        results[vec_name] = {}
        cv_results[vec_name] = {}
        for clf_name, clf in classifiers.items():
            # Fresh copies
            from sklearn.base import clone
            vec_c = clone(vec)
            clf_c = clone(clf)

            acc, prec, rec, f1, cm, y_pred = evaluate(clf_c, vec_c, X_train, X_test, y_train, y_test)
            cv_mean, cv_std = cross_val(clone(clf), clone(vec), X, y)

            results[vec_name][clf_name] = {
                "acc": acc, "prec": prec, "rec": rec, "f1": f1, "cm": cm
            }
            cv_results[vec_name][clf_name] = (cv_mean, cv_std)

            print(f"  {vec_name:20s} | {clf_name:22s} | Acc={acc:.3f} F1={f1:.3f} CV={cv_mean:.3f}±{cv_std:.3f}")

        # Fit vectoriser for feature importance on BOW / TF-IDF Unigram / TF-IDF Bigram
        vc = clone(vec)
        X_tr_fit = vc.fit_transform(X_train)
        vf_data[vec_name] = (vc, X_tr_fit, y_train)

    # ── Plots ──
    print("\n[Generating Plots]")
    plot_dataset_dist(df, outdir)
    plot_confusion_matrices(results, outdir)
    plot_metrics_comparison(results, outdir)
    plot_cross_val(cv_results, outdir)
    plot_top_features(vf_data, outdir)

    # ── Summary CSV ──
    rows = []
    for vec_name, clf_dict in results.items():
        for clf_name, m in clf_dict.items():
            cv_m, cv_s = cv_results[vec_name][clf_name]
            rows.append({
                "Feature Representation": vec_name,
                "Classifier": clf_name,
                "Accuracy": round(m["acc"], 4),
                "Precision": round(m["prec"], 4),
                "Recall": round(m["rec"], 4),
                "F1 Score": round(m["f1"], 4),
                "CV Mean": round(cv_m, 4),
                "CV Std": round(cv_s, 4),
            })
    df_results = pd.DataFrame(rows)
    csv_path = os.path.join(outdir, "results_summary.csv")
    df_results.to_csv(csv_path, index=False)
    print(f"  [✓] results_summary.csv saved")

    print("\n[Final Results Summary]")
    print(df_results.to_string(index=False))

    # Best model
    best = df_results.sort_values("F1 Score", ascending=False).iloc[0]
    print(f"\n  ★  Best: {best['Classifier']} + {best['Feature Representation']} → F1={best['F1 Score']:.4f}")

    # ── Save dataset ──
    df.to_csv(os.path.join(outdir, "dataset.csv"), index=False)
    print(f"  [✓] dataset.csv saved")

    return df_results, results

if __name__ == "__main__":
    df_results, results = main()
