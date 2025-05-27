
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog, messagebox
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Funções de ativação e suas derivadas
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivada(x):
    return x * (1 - x)

def tanh(x):
    return np.tanh(x)

def tanh_derivada(x):
    return 1 - np.power(x, 2)

def linear(x):
    return x / 10

def linear_derivada(x):
    return 0.1

# Carregar CSV
def carregar_dados(path, ativacao):
    df = pd.read_csv(path)

    entradas = df.iloc[:, :-1].values
    saidas_raw = df.iloc[:, -1]
    classes = sorted(saidas_raw.unique())

    if ativacao == 'linear':
        # Usar dados originais ou padronizados (z-score)
        entradas_corrigidas = (entradas - entradas.mean(axis=0)) / entradas.std(axis=0)
    else:
        # Usar normalização Min-Max (0-1) para sigmoid/tanh
        entradas_corrigidas = (entradas - entradas.min(axis=0)) / (entradas.max(axis=0) - entradas.min(axis=0))

    # Codificação one-hot para classes
    saidas = np.zeros((len(saidas_raw), len(classes)))
    for i, classe in enumerate(saidas_raw):
        saidas[i, classes.index(classe)] = 1

    return entradas_corrigidas, saidas, classes, saidas_raw.values


# Inicialização dos pesos

def inicializar_pesos(entrada_dim, oculta_dim, saida_dim):
    pesos_entrada_oculta = np.random.uniform(-0.1, 0.1, (entrada_dim, oculta_dim))
    pesos_oculta_saida = np.random.uniform(-0.1, 0.1, (oculta_dim, saida_dim))
    return pesos_entrada_oculta, pesos_oculta_saida


# Gráfico de erro por época
def plot_erro_por_epoca(erros):
    plt.figure(figsize=(10, 5))
    plt.plot(erros, label='Erro médio por época')
    plt.xlabel('Época')
    plt.ylabel('Erro médio')
    plt.title('Evolução do erro no treinamento')
    plt.legend()
    plt.grid(True)
    plt.show()

# Matriz de Confusão
def mostrar_matriz_confusao(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Matriz de Confusão")
    plt.grid(False)
    plt.show()

# Treinamento da rede
ativacoes = {
    'sigmoid': (sigmoid, sigmoid_derivada),
    'tanh': (tanh, tanh_derivada),
    'linear': (linear, linear_derivada)
}

def treinar_rede(X, y, n_oculta, taxa_aprendizado=0.1, max_epocas=1000, erro_limite=None, ativacao='sigmoid'):
    func, dfunc = ativacoes[ativacao]
    n_entrada = X.shape[1]
    n_saida = y.shape[1]

    W1, W2 = inicializar_pesos(n_entrada, n_oculta, n_saida)
    erros_por_epoca = []

    for epoca in range(max_epocas):
        erros_amostrais = []

        for i in range(X.shape[0]):
            entrada = X[i].reshape(1, -1)
            esperado = y[i].reshape(1, -1)

            # Forward
            net_oculta = np.dot(entrada, W1)
            saida_oculta = func(net_oculta)
            net_saida = np.dot(saida_oculta, W2)
            saida_final = func(net_saida)

            # Cálculo dos erros e backpropagation
            erro_saida = (esperado - saida_final) * dfunc(saida_final)
            erro_oculta = np.dot(erro_saida, W2.T) * dfunc(saida_oculta)

            W2 += taxa_aprendizado * np.dot(saida_oculta.T, erro_saida)
            W1 += taxa_aprendizado * np.dot(entrada.T, erro_oculta)

          
            erro_amostra = 0.5 * np.sum((esperado - saida_final) ** 2)

            erros_amostrais.append(erro_amostra)

       
        erro_medio = np.mean(erros_amostrais)
        erros_por_epoca.append(erro_medio)

        if erro_limite is not None and erro_medio < erro_limite:
            print(f"Critério de erro atingido na época {epoca}: erro médio = {erro_medio:.6f}")
            break

    return W1, W2, erros_por_epoca


# Variáveis globais para armazenar pesos e classes treinados
W1_trained = None
W2_trained = None
classes_trained = None
ativacao_trained = None

def iniciar_interface():
    def selecionar_arquivo_treinamento():
        caminho = filedialog.askopenfilename(title="Selecione o arquivo CSV de treinamento", filetypes=[("CSV files", "*.csv")])
        entrada_csv.set(caminho)

    def selecionar_arquivo_teste():
        caminho = filedialog.askopenfilename(title="Selecione o arquivo CSV de teste", filetypes=[("CSV files", "*.csv")])
        entrada_csv_teste.set(caminho)

    def iniciar_treinamento():
        global W1_trained, W2_trained, classes_trained, ativacao_trained
        try:
            ativacao = var_ativacao.get()  
            entradas, saidas, classes, classes_reais = carregar_dados(entrada_csv.get(), ativacao)

            n_oculta = int(entrada_neuronios.get())
            taxa = float(entrada_taxa.get())
            max_epocas = int(entrada_epocas.get())
            erro = float(entrada_erro.get())
           

            W1, W2, erros = treinar_rede(entradas, saidas, n_oculta, taxa, max_epocas, erro, ativacao)
            plot_erro_por_epoca(erros)

            # Armazenar para teste futuro
            W1_trained, W2_trained, classes_trained, ativacao_trained = W1, W2, classes, ativacao

            # Habilitar botão de teste
            botao_selecionar_teste.config(state=tk.NORMAL)
            botao_testar.config(state=tk.NORMAL)

            messagebox.showinfo("Treinamento", "Rede treinada com sucesso! Selecione o arquivo de teste e clique em Testar Rede.")
        except Exception as e:
            messagebox.showerror("Erro", str(e))

    def testar_rede():
        try:
            if W1_trained is None:
                raise Exception("A rede ainda não foi treinada!")
            entradas_teste, _, _, classes_reais_teste = carregar_dados(entrada_csv_teste.get(), ativacao_trained)


            func, _ = ativacoes[ativacao_trained]
            saidas_ocultas = func(np.dot(entradas_teste, W1_trained))
            saidas_finais = func(np.dot(saidas_ocultas, W2_trained))
            predicoes = [classes_trained[np.argmax(linha)] for linha in saidas_finais]

            mostrar_matriz_confusao(classes_reais_teste, predicoes, labels=classes_trained)
            
            # Calcular e mostrar acurácia
            total = len(classes_reais_teste)
            acertos = sum([1 for i in range(total) if classes_reais_teste[i] == predicoes[i]])
            acuracia = acertos / total * 100
            messagebox.showinfo("Resultados", f"Total de exemplos: {total}\nAcertos: {acertos}\nAcurácia: {acuracia:.2f}%")
        except Exception as e:
            messagebox.showerror("Erro", str(e))

    app = tk.Tk()
    app.title("Configuração da Rede Neural MLP")

    tk.Label(app, text="Arquivo CSV de Treinamento:").grid(row=0, column=0)
    entrada_csv = tk.StringVar()
    tk.Entry(app, textvariable=entrada_csv, width=40).grid(row=0, column=1)
    tk.Button(app, text="Selecionar", command=selecionar_arquivo_treinamento).grid(row=0, column=2)

    tk.Label(app, text="Neurônios na camada oculta:").grid(row=1, column=0)
    entrada_neuronios = tk.Entry(app)
    entrada_neuronios.grid(row=1, column=1)

    tk.Label(app, text="Taxa de Aprendizado (0-1):").grid(row=2, column=0)
    entrada_taxa = tk.Entry(app)
    entrada_taxa.grid(row=2, column=1)

    tk.Label(app, text="Máximo de Épocas:").grid(row=3, column=0)
    entrada_epocas = tk.Entry(app)
    entrada_epocas.grid(row=3, column=1)

    tk.Label(app, text="Erro mínimo desejado:").grid(row=4, column=0)
    entrada_erro = tk.Entry(app)
    entrada_erro.grid(row=4, column=1)

    tk.Label(app, text="Função de Ativação:").grid(row=5, column=0)
    var_ativacao = tk.StringVar(value='sigmoid')
    tk.OptionMenu(app, var_ativacao, 'sigmoid', 'tanh', 'linear').grid(row=5, column=1)

    tk.Button(app, text="Iniciar Treinamento", command=iniciar_treinamento).grid(row=6, column=0, columnspan=3)

    # Novos elementos para teste
    tk.Label(app, text="Arquivo CSV de Teste:").grid(row=7, column=0)
    entrada_csv_teste = tk.StringVar()
    tk.Entry(app, textvariable=entrada_csv_teste, width=40).grid(row=7, column=1)
    botao_selecionar_teste = tk.Button(app, text="Selecionar", command=selecionar_arquivo_teste, state=tk.DISABLED)
    botao_selecionar_teste.grid(row=7, column=2)

    botao_testar = tk.Button(app, text="Testar Rede", command=testar_rede, state=tk.DISABLED)
    botao_testar.grid(row=8, column=0, columnspan=3)

    app.mainloop()


if __name__ == '__main__':
    iniciar_interface()
