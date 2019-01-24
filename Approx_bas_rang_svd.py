import csv
import numpy as np


def open_numeric(baseFileName, fieldnames=['user', 'movie', 'rating', 'datestamp'], delimiter='\t'):
    """
    fonction généralisée pour lire les fichiers numériques (bases et tests)
    """
    with open(baseFileName, 'r') as f:
         reader = csv.DictReader(f, delimiter = delimiter, fieldnames=fieldnames)
         # create a dict out of reader, converting all values to integers
         return [dict([key, int(value)] for key, value in row.items()) for row in list(reader)]


def creation_RC(R):
    """
    :param R: matrice R utilisateur-item
    :return:  matrice RC des moyennes des films
    """
    NbUsers = R.shape[0]
    NbItems = R.shape[1]
    mean_items = [0 for row in range(NbItems)]  # liste des moyenne par colonne(film)
    for col in range(NbItems):
        rating_number = 0  # savoir le nombre d'éléments quon additionne pour diviser la moyenne
        for user_rating in R[:, col]:
            if user_rating != 0:  # on ne prend que les valeurs non nul
                mean_items[col] += user_rating
                rating_number += 1
        if rating_number != 0:  # il doit y avoir au moins une note par film
            mean_items[col] = mean_items[col] / rating_number
    RC = R.copy()
    for col in range(NbItems):
        for row in range(NbUsers):
            if RC[row, col] == 0:
                RC[row, col] = mean_items[col]
    print("Shape de RC : ", RC.shape)
    return RC


def creation_RR(R):
    """
    :param R: matrice R utilisateur-item
    :return: matrice RC des moyennes utilisateurs
    """
    NbUsers = R.shape[0]
    NbItems = R.shape[1]
    mean_users = [0 for row in range(NbUsers)]  # liste des moyenne par lignes(users)
    for row in range(NbUsers):
        rating_number = 0  # savoir le nombre d'éléments qu'on additionne pour diviser la moyenne
        for user_rating in R[row, :]:
            if user_rating != 0:
                mean_users[row] += user_rating
                rating_number += 1
        if rating_number != 0:  # il doit y avoir au moins une note par film
            mean_users[row] = mean_users[row] / rating_number
    RR = R.copy()
    for row in range(NbUsers):
        for col in range(NbItems):
            if RR[row, col] == 0:
                RR[row, col] = mean_users[row]
    print("Shape de RR : ", RR.shape)
    return RR


def svd_matrice(A):
    """
    A : matrice de taille MxN (ici M < N)
    Fonction qui réalise la décomposition en valeurs singulières de A tq : A = U x Sigma x VT
    Retourne :
    U : matrice de taille MxM unitaire
    Sigma : matrice  diagonal de taille MxN
    VT : Transpose de la matrice V qui est de taille NxN unitaire
    """
    U, s, VT = np.linalg.svd(A, full_matrices=True)   # s est un vecteur de taille M (M < N) contenant les coeffs
    S = np.zeros((A.shape[0], A.shape[1])) # on créé S une matrice de taille M x N
    m = min(A.shape[0], A.shape[1]) # ici M < N donc on aura m = M
    S[:m, :m] = np.diag(s) # On rempli S par la matrice carre diagonale MxM contenant les coeffs
    return U, S, VT


def approx_rang_k_svd_(U, S, VT):
    """
    :param U: matrice U résultant de la svd
    :param S: matrice S résultant de la svd
    :param VT: matrice VT résultant de la svd
    :return: un tuple contenant les matrices Uk, Sk, VTk pour k = 1,..,30
    """
    liste_Sk = []
    liste_Uk = []
    liste_VTk = []
    liste_Rk = []
    for k in range(1, 31):
        Sk = S[:k, :k]  # on remplie Sck (matrice carrée diagonale remplie des k premiers coeffs)
        liste_Sk.append(Sk)
        # matrice de rang k
        Uk = U[:, :k]
        VTk = VT[:k, :]
        Rk = np.dot(Uk, np.dot(Sk, VTk))
        liste_Uk.append(Uk)
        liste_VTk.append(VTk)
        liste_Rk.append(Rk)
    return liste_Rk, liste_Uk, liste_Sk, liste_VTk


def calcul_matrices_prediction(liste_Uk, liste_Sk, liste_VTk):
    """
    :param liste_Uk: liste des matrices Uk
    :param liste_Sk: liste des matrices Vk
    :param liste_VTk: liste des matrices VTk
    :return: une liste des matrices de prédictions pour k = 1,..,30
    """
    liste_matrices_predictions = []
    for k in range(0, 30):
        matrice_predictions = np.dot(liste_Uk[k], np.dot(liste_Sk[k], liste_VTk[k]))
        liste_matrices_predictions.append(matrice_predictions)
    return liste_matrices_predictions


def calcul_MAE(Rtest, liste_matrices_prediction):
    """
    Calcule la MAE pour chaque matrice de prédiction
    :param Rtest: Matrice Test
    :param liste_matrices_prediction: liste des matrices de prédictions
    :return: liste des MAE pour ces matrices de matrices de prédictions
    """
    liste_MAE = []
    for k in range(0, 30):
        errorRating = []
        for i in range(0, Rtest.shape[0]):
            for j in range(0, Rtest.shape[1]):
                if Rtest[i][j] != 0:
                    errorRating.append(liste_matrices_prediction[k][i][j] - Rtest[i][j])
        liste_MAE.append(np.mean(np.abs(errorRating)))
    return liste_MAE


def meilleure_matrice(liste_MAE):
    """
    Trouve quelle matrice donne les meilleures predictions et à quel K elle correspond
    :param liste_MAE: liste contenant les MAE des 30 matrices
    :return: la MAE la plus faible et pour quelle approximation de rang k on l'obtient
    """
    min_MAE = liste_MAE[0]
    index_min = 1
    for k in range(len(liste_MAE)):
        if liste_MAE[k] < min_MAE:
            min_MAE = liste_MAE[k]
            index_min = k + 1
    return min_MAE, index_min


def general_function(numero_base_test):
    """
    numero_base_test : numero du set de base/test que l'on soouhaite utilisé
    return : La plus petite MAE et pour quel K on a donc obtenu les meilleures predictions
    """
    baseFile = "u" + str(numero_base_test) + ".base"
    testFile = "u" + str(numero_base_test) + ".test"
    baseUserItem = open_numeric(baseFile)
    testUserItem = open_numeric(testFile)

    # Matrice Test (pour évaluer la qualité de nos  prédictions)
    NbUsers = 943
    NbItems = 1682
    Rtest = np.zeros((NbUsers, NbItems))
    for row in testUserItem:
        Rtest[row['user'] - 1, row['movie'] - 1] = row['rating']

    # Remplir la matrice utilisateur-item R
    R = np.zeros((NbUsers, NbItems))
    for row in baseUserItem:
        R[row['user'] - 1, row['movie'] - 1] = row['rating']

    # Remplissage matrice RC (note moyenne film)
    RC = creation_RC(R)

    # Remplissage matrice RR (note moyenne user)
    RR = creation_RR(R)

    # On fait les svd :
    Uc, Sc, VTc = svd_matrice(RC)
    print("A-t-on bien RC = Uc x Sc x VTc : ", np.allclose(RC, np.dot(Uc, np.dot(Sc, VTc))))
    Ur, Sr, VTr = svd_matrice(RR)
    print("A-t-on bien RR = Ur x Sr x VTr : ", np.allclose(RR, np.dot(Ur, np.dot(Sr, VTr))))

    # Approximation de rang K (k = 1, .., 31) de la matrice RC
    liste_RCk, liste_Uck, liste_Sck, liste_VTck = approx_rang_k_svd_(Uc, Sc, VTc)

    # Approximation de rang K (k = 1, .., 31) de la matrice RR
    liste_RRk, liste_Urk, liste_Srk, liste_VTrk = approx_rang_k_svd_(Ur, Sr, VTr)

    print("Les approximations ont été calculées")

    # Calcul des matrices de predictions pour k = 1,...,30 :
    liste_matrices_predictions_RC = calcul_matrices_prediction(liste_Uck, liste_Sck, liste_VTck)
    liste_matrices_predictions_RR = calcul_matrices_prediction(liste_Urk, liste_Srk, liste_VTrk)

    print("Les matrices de prédictions ont été calculées")

    # Calcul des MAE :
    liste_MAEc = calcul_MAE(Rtest, liste_matrices_predictions_RC)
    liste_MAEr = calcul_MAE(Rtest, liste_matrices_predictions_RR)

    # Pour quel K a-t-on les meilleures prédictions :
    min_MAEc, index_minC = meilleure_matrice(liste_MAEc)
    print(" Pour RC : La meilleure matrice de prédiction a une MAE = ", min_MAEc,
          " et est obtenue pour l'approximation de rang k = ", index_minC)

    min_MAEr, index_minR = meilleure_matrice(liste_MAEr)
    print(" Pour RR : La meilleure matrice de prédiction a une MAE = ", min_MAEr,
          " et est obtenue pour l'approximation de rang k = ", index_minR)

   return min_MAEc,index_minC, min_MAEr, index_minR


for i in range(1, 6):
    general_function(i)
