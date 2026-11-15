
# Implementacao usando apenas a biblioteca padrao do Python
# Nota-se que as bibliotecas não aplicam a solução diretamente, mas são usadas para auxiliar em algumas operações,
# como manipulação de strings e cálculos.

import math
import re
import unicodedata


# 1. BASE DE DADOS
# Cada documento e um texto curto que sera indexado pelo sistema.

DOCUMENTOS = [
    "A manga e uma fruta tropical doce e madura.",
    "A manga pode ser consumida em sucos e sobremesas.",
    "A banana e uma fruta rica em potassio.",
    "A maca e uma fruta usada em tortas e sobremesas.",
    "A manga da camisa esta rasgada na costura.",
    "A manga comprida protege o braco do frio.",
    "A camiseta possui manga curta e gola redonda.",
    "A costureira ajustou a manga do vestido.",
    "O suco de manga e uma bebida feita com fruta fresca.",
    "A fruta manga amadurece em regioes de clima quente.",
    "Uma salada de frutas pode incluir manga, banana e maca.",
    "A blusa de manga longa e adequada para o inverno.",
    "A costura da manga da jaqueta precisa de reparo.",
    "A camisa social tem mangas compridas e punhos.",
    "O suco de manga gelado é excelente para os dias quentes de verão.",
    "Comprei uma camiseta nova de manga curta para passear no parque.",
    "A receita do bolo de frutas leva pedaços de banana e maçã.",
    "Gosto de comer manga com iogurte natural no café da manhã.",
    "A costureira recomendou ajustar a manga da camiseta para que ficasse perfeita."
]

# 2. PRE-PROCESSAMENTO DOS TEXTOS
# Normaliza acentos, transforma em minusculas e remove palavras
# muito comuns (stopwords), que geralmente pouco ajudam na busca.

STOPWORDS = {
    "a", "o", "as", "os", "um", "uma", "uns", "umas",
    "de", "do", "da", "dos", "das", "em", "no", "na",
    "nos", "nas", "e", "ou", "que", "para", "por",
    "com", "ao", "aos", "esta", "estao", "ser", "sao",
    "foi", "pode", "ser", "se", "sua", "seu"
}

def remover_acentos(texto):
    """Retira acentos para padronizar palavras."""
    texto_normalizado = unicodedata.normalize("NFD", texto)
    return "".join(
        caractere for caractere in texto_normalizado
        if unicodedata.category(caractere) != "Mn"
    )

def tokenizar(texto):
    """Converte o texto em uma lista de palavras relevantes."""
    texto = remover_acentos(texto.lower())
    palavras = re.findall(r"[a-z0-9]+", texto)
    return [palavra for palavra in palavras if palavra not in STOPWORDS]


# 3. CONSTRUCAO DA MATRIZ TERMO-DOCUMENTO
# Cada linha representa um termo e cada coluna representa
# um documento. O valor e a frequencia do termo no documento.

def construir_matriz(documentos):
    """Retorna vocabulario, matriz termo-documento e documentos tokenizados."""
    documentos_tokens = [tokenizar(doc) for doc in documentos]

    vocabulario = sorted({
        palavra
        for tokens in documentos_tokens
        for palavra in tokens
    })

    matriz = []
    for termo in vocabulario:
        linha = []
        for tokens in documentos_tokens:
            linha.append(tokens.count(termo))
        matriz.append(linha)

    return vocabulario, matriz, documentos_tokens


# 4. OPERACOES MATRICIAIS BASICAS
# Estas funcoes substituem operacoes que seriam feitas por
# bibliotecas numericas prontas.


def transposta(matriz):
    """Calcula a transposta de uma matriz."""
    return [list(linha) for linha in zip(*matriz)]


def multiplicar_matrizes(A, B):
    """Multiplica duas matrizes compativeis."""
    if not A or not B or len(A[0]) != len(B):
        raise ValueError("Dimensoes incompativeis para multiplicacao.")

    resultado = []
    BT = transposta(B)

    for linha in A:
        nova_linha = []
        for coluna in BT:
            nova_linha.append(sum(x * y for x, y in zip(linha, coluna)))
        resultado.append(nova_linha)

    return resultado


def produto_interno(vetor_a, vetor_b):
    """Calcula o produto interno entre dois vetores."""
    return sum(a * b for a, b in zip(vetor_a, vetor_b))


def norma(vetor):
    """Calcula a norma euclidiana de um vetor."""
    return math.sqrt(sum(valor * valor for valor in vetor))


def similaridade_cosseno(vetor_a, vetor_b):
    """Calcula a similaridade do cosseno entre dois vetores."""
    norma_a = norma(vetor_a)
    norma_b = norma(vetor_b)

    if norma_a == 0 or norma_b == 0:
        return 0.0

    return produto_interno(vetor_a, vetor_b) / (norma_a * norma_b)


# 5. AUTOVALORES E AUTOVETORES PELO METODO DE JACOBI
# O metodo de Jacobi diagonaliza uma matriz simetrica real.
# Aqui sera aplicado a A^T * A, que e simetrica.

def jacobi_autovalores(matriz, tolerancia=1e-10, max_iteracoes=10000):
    """Calcula autovalores e autovetores de uma matriz simetrica."""
    n = len(matriz)
    A = [linha[:] for linha in matriz]

    # Inicializa a matriz de autovetores como identidade.
    V = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]

    # Executa rotacoes ate que os elementos fora da diagonal sejam pequenos.
    for _ in range(max_iteracoes):
        maior = 0.0
        p, q = 0, 0

        # Localiza o maior elemento fora da diagonal principal.
        for i in range(n):
            for j in range(i + 1, n):
                if abs(A[i][j]) > maior:
                    maior = abs(A[i][j])
                    p, q = i, j

        # Se a matriz ja esta praticamente diagonal, encerra.
        if maior < tolerancia:
            break

        # Calcula o angulo da rotacao de Jacobi.
        if abs(A[p][q]) < tolerancia:
            continue

        tau = (A[q][q] - A[p][p]) / (2.0 * A[p][q])
        sinal = 1.0 if tau >= 0 else -1.0
        t = sinal / (abs(tau) + math.sqrt(1.0 + tau * tau))
        c = 1.0 / math.sqrt(1.0 + t * t)
        s = t * c

        app = A[p][p]
        aqq = A[q][q]
        apq = A[p][q]

        # Atualiza os elementos da matriz simetrica.
        A[p][p] = app - t * apq
        A[q][q] = aqq + t * apq
        A[p][q] = 0.0
        A[q][p] = 0.0

        # Atualiza as demais posicoes das linhas e colunas p e q.
        for r in range(n):
            if r != p and r != q:
                arp = A[r][p]
                arq = A[r][q]
                A[r][p] = c * arp - s * arq
                A[p][r] = A[r][p]
                A[r][q] = s * arp + c * arq
                A[q][r] = A[r][q]

        # Aplica a mesma rotacao aos autovetores.
        for r in range(n):
            vrp = V[r][p]
            vrq = V[r][q]
            V[r][p] = c * vrp - s * vrq
            V[r][q] = s * vrp + c * vrq

    # Os elementos da diagonal sao os autovalores.
    autovalores = [A[i][i] for i in range(n)]

    # Ordena autovalores em ordem decrescente e reorganiza autovetores.
    indices = sorted(range(n), key=lambda i: autovalores[i], reverse=True)
    valores_ordenados = [max(0.0, autovalores[i]) for i in indices]
    vetores_ordenados = [
        [V[linha][i] for i in indices]
        for linha in range(n)
    ]

    return valores_ordenados, vetores_ordenados



# 6. DECOMPOSICAO SVD
# Para A (termos x documentos), calculamos A^T*A.
# Seus autovetores formam V; as raizes dos autovalores sao
# os valores singulares. U e calculada por A*V/sigma.


def decomposicao_svd(A):
    """Calcula uma decomposicao SVD didatica de A."""
    if not A or not A[0]:
        raise ValueError("A matriz nao pode ser vazia.")

    AT = transposta(A)
    ATA = multiplicar_matrizes(AT, A)

    autovalores, V = jacobi_autovalores(ATA)

    # O numero de componentes singulares nao pode exceder min(m, n).
    m = len(A)
    n = len(A[0])
    quantidade = min(m, n)

    valores_singulares = [
        math.sqrt(max(0.0, valor))
        for valor in autovalores[:quantidade]
    ]

    # Mantem somente as colunas correspondentes aos valores singulares.
    V_reduzida = [linha[:quantidade] for linha in V]

    # Calcula U = A*V/sigma para cada valor singular nao nulo.
    AV = multiplicar_matrizes(A, V_reduzida)
    U = [[0.0 for _ in range(quantidade)] for _ in range(m)]

    for j in range(quantidade):
        sigma = valores_singulares[j]
        if sigma > 1e-12:
            for i in range(m):
                U[i][j] = AV[i][j] / sigma

    # Converte V para V transposta.
    VT = transposta(V_reduzida)

    # Constroi Sigma como matriz diagonal.
    Sigma = [
        [valores_singulares[i] if i == j else 0.0
         for j in range(quantidade)]
        for i in range(quantidade)
    ]

    return U, Sigma, VT, valores_singulares


# 7. PROJECAO NO ESPACO REDUZIDO
# A consulta e projetada por U^T*q.
# Cada documento e representado por Sigma*V^T.

def projetar_consulta(U, consulta_vetor, k):
    """Projeta a consulta nas k dimensoes latentes."""
    UT = transposta(U)
    resultado = []

    for i in range(k):
        resultado.append(produto_interno(UT[i], consulta_vetor))

    return resultado


def coordenadas_documentos(Sigma, VT, k):
    """Calcula as coordenadas dos documentos no espaco LSI."""
    quantidade_documentos = len(VT[0])
    coordenadas = []

    for doc in range(quantidade_documentos):
        vetor = []
        for dimensao in range(k):
            vetor.append(Sigma[dimensao][dimensao] * VT[dimensao][doc])
        coordenadas.append(vetor)

    return coordenadas

# 8. VETOR DA CONSULTA
# Usa o mesmo vocabulario da base para transformar a consulta
# em um vetor numerico compativel com a matriz termo-documento.


def vetorizar_consulta(consulta, vocabulario):
    """Transforma a consulta em um vetor de frequencias."""
    tokens = tokenizar(consulta)
    return [tokens.count(termo) for termo in vocabulario]



# 9. RECUPERACAO E COMPARACAO DOS RESULTADOS
# Calcula similaridade no espaco original e no espaco LSI.


def ranquear_documentos(vetores_documentos, vetor_consulta, documentos):
    """Ordena documentos pela similaridade do cosseno."""
    resultados = []

    for indice, vetor_doc in enumerate(vetores_documentos):
        nota = similaridade_cosseno(vetor_consulta, vetor_doc)
        resultados.append((indice, nota, documentos[indice]))

    return sorted(resultados, key=lambda item: item[1], reverse=True)


def executar_busca(consulta, documentos=DOCUMENTOS, k=2, quantidade_resultados=3):
    """Executa busca tradicional e busca com reducao dimensional LSI."""
    vocabulario, A, _ = construir_matriz(documentos)

    # Cria o vetor da consulta na mesma base de termos.
    q = vetorizar_consulta(consulta, vocabulario)

    # Busca sem reducao: compara a consulta com as colunas originais de A.
    documentos_originais = transposta(A)
    resultados_originais = ranquear_documentos(
        documentos_originais, q, documentos
    )

    # Calcula a decomposicao SVD da matriz termo-documento.
    U, Sigma, VT, valores_singulares = decomposicao_svd(A)

    # Limita k ao numero de componentes disponiveis.
    k = max(1, min(k, len(valores_singulares)))

    # Projeta consulta e documentos no espaco semantico reduzido.
    q_reduzido = projetar_consulta(U, q, k)
    docs_reduzidos = coordenadas_documentos(Sigma, VT, k)

    # Busca com reducao dimensional.
    resultados_lsi = ranquear_documentos(
        docs_reduzidos, q_reduzido, documentos
    )

    return {
        "vocabulario": vocabulario,
        "matriz": A,
        "valores_singulares": valores_singulares,
        "resultados_originais": resultados_originais[:quantidade_resultados],
        "resultados_lsi": resultados_lsi[:quantidade_resultados],
        "k": k
    }


# 10. EXIBICAO DOS RESULTADOS
# Mostra os documentos mais relevantes e suas pontuacoes.

def exibir_resultados(titulo, resultados):
    """Imprime os resultados de uma busca."""
    print("\n" + titulo)
    print("-" * len(titulo))

    for posicao, (indice, nota, documento) in enumerate(resultados, start=1):
        print(f"{posicao}. Documento {indice + 1} | Similaridade: {nota:.4f}")
        print(f"   {documento}")


def exibir_relatorio(resultado):
    """Exibe comparacao entre busca tradicional e busca LSI."""
    print("\n========== RELATORIO ANALITICO ==========")
    print(f"Dimensoes da matriz: {len(resultado['matriz'])} x "
          f"{len(resultado['matriz'][0])}")
    print(f"Quantidade de dimensoes mantidas (k): {resultado['k']}")
    print("Valores singulares:",
          [round(valor, 4) for valor in resultado["valores_singulares"]])

    exibir_resultados(
        "BUSCA SEM REDUCAO DIMENSIONAL",
        resultado["resultados_originais"]
    )
    exibir_resultados(
        "BUSCA COM LSI / SVD",
        resultado["resultados_lsi"]
    )



# 11. PROGRAMA PRINCIPAL
# Permite executar exemplos automaticos ou receber uma consulta.

def main():
    """Menu principal: permite consultar ou executar os testes."""
    while True:
        print("\nSISTEMA DE RECUPERACAO DE INFORMACAO - LSI")
        print("1 - Fazer uma consulta")
        print("2 - Executar os dois casos de teste")
        print("0 - Encerrar")
        opcao = input("Escolha: ").strip()

        if opcao == "1":
            consulta = input("Digite as palavras-chave: ").strip()
            if consulta:
                resultado = executar_busca(consulta, k=3)
                exibir_relatorio(resultado)
            else:
                print("A consulta nao pode ficar vazia.")
        elif opcao == "2":
            executar_testes()
        elif opcao == "0":
            print("Programa encerrado.")
            break
        else:
            print("Opcao invalida. Escolha 1, 2 ou 0.")


# 12. TESTES AUTOMATICOS
# Os dois casos abaixo podem ser executados para demonstracao.

def executar_testes():
    """Executa dois casos de teste predefinidos."""
    casos = [
        "manga fruta madura suco",
        "manga camisa costura roupa"
    ]

    for numero, consulta in enumerate(casos, start=1):
        print("\n\n========== CASO DE TESTE", numero, "==========")
        print("Entrada:", consulta)
        resultado = executar_busca(consulta, k=3)
        exibir_relatorio(resultado)


if __name__ == "__main__":
    # Para testar automaticamente, troque main() por executar_testes().
    main()
