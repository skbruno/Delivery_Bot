import matplotlib.pyplot as plt
import numpy as np

# Dados fornecidos
pontuacao_com_ordem = 37
pontuacao_sem_ordem = 93
passos_com_ordem = 163
passos_sem_ordem = 103

# Configurações do gráfico
algoritmos = ['Com Ordem', 'Sem Ordem']
cores = ['#1f77b4', '#ff7f0e']
largura_barra = 0.35

# Criando figura com dois subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
fig.suptitle('Comparação entre Algoritmo com Ordem e Sem Ordem', fontsize=14, fontweight='bold')

# Gráfico de Pontuação
ax1.bar(algoritmos, [pontuacao_com_ordem, pontuacao_sem_ordem], color=cores, width=largura_barra)
ax1.set_title('Pontuação', pad=20)
ax1.set_ylabel('Pontos')
ax1.set_ylim(0, 100)

# Adicionando valores nas barras
for i, valor in enumerate([pontuacao_com_ordem, pontuacao_sem_ordem]):
    ax1.text(i, valor + 2, str(valor), ha='center', va='bottom', fontweight='bold')

# Gráfico de Passos
ax2.bar(algoritmos, [passos_com_ordem, passos_sem_ordem], color=cores, width=largura_barra)
ax2.set_title('Passos Executados', pad=20)
ax2.set_ylabel('Número de Passos')
ax2.set_ylim(0, max(passos_com_ordem, passos_sem_ordem) + 20)

# Adicionando valores nas barras
for i, valor in enumerate([passos_com_ordem, passos_sem_ordem]):
    ax2.text(i, valor + 5, str(valor), ha='center', va='bottom', fontweight='bold')

# Ajustando layout
plt.tight_layout()
plt.subplots_adjust(top=0.85, wspace=0.3)

# Mostrando o gráfico
plt.show()