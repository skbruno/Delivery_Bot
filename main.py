import pygame
import random
import heapq
import sys
import argparse
from abc import ABC, abstractmethod

# ==========================
# CLASSES DE PLAYER
# ==========================
class BasePlayer(ABC):
    """
    Classe base para o jogador (robô).
    Para criar uma nova estratégia de jogador, basta herdar dessa classe e implementar o método escolher_alvo.
    """
    def __init__(self, position):
        self.position = position  # Posição no grid [x, y]
        self.cargo = 0            # Número de pacotes atualmente carregados
        self.battery = 70         # Nível da bateria

    @abstractmethod
    def escolher_alvo(self, world):
        """
        Retorna o alvo (posição) que o jogador deseja ir.
        Recebe o objeto world para acesso a pacotes e metas.
        """
        pass

class DefaultPlayer(BasePlayer):
    """
    Implementação padrão do jogador.
    Coleta todos os pacotes primeiro (usando rota otimizada com A*),
    depois entrega todos (usando rota otimizada com A*).
    """
    def __init__(self, position):
        super().__init__(position)
        self.collection_phase = True  # Fase de coleta (True) ou entrega (False)
        self.full_path = []           # Caminho completo a ser seguido
        self.current_target_index = 0 # Índice do alvo atual no caminho
        self.need_replan = True       # Flag para indicar quando replanejar
        
    def find_optimal_route(self, points, world):
        """Encontra a rota mais curta que visita todos os pontos usando heurística de vizinho mais próximo"""
        if not points:
            return []
            
        # Começa na posição atual do jogador
        current_pos = self.position
        unvisited = points.copy()
        route = []
        
        while unvisited:
            # Encontra o ponto mais próximo do ponto atual
            nearest = min(unvisited, key=lambda p: self.heuristic(current_pos, p))
            # Calcula o caminho até ele
            path = self.astar(current_pos, nearest, world)
            if not path:
                break
            # Adiciona ao caminho total (exceto a primeira posição que é a atual)
            route.extend(path)
            # Atualiza a posição atual
            current_pos = nearest
            # Remove o ponto da lista de não visitados
            unvisited.remove(nearest)
            
        return route
    
    def heuristic(self, a, b):
        """Distância de Manhattan"""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
        
    def astar(self, start, goal, world):
        """Algoritmo A* para encontrar caminho entre dois pontos"""
        maze = world.map
        size = world.maze_size
        neighbors = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        close_set = set()
        came_from = {}
        gscore = {tuple(start): 0}
        fscore = {tuple(start): self.heuristic(start, goal)}
        oheap = []
        heapq.heappush(oheap, (fscore[tuple(start)], tuple(start)))
        
        while oheap:
            current = heapq.heappop(oheap)[1]
            if list(current) == goal:
                data = []
                while current in came_from:
                    data.append(list(current))
                    current = came_from[current]
                data.reverse()
                return data
                
            close_set.add(current)
            for dx, dy in neighbors:
                neighbor = (current[0] + dx, current[1] + dy)
                tentative_g = gscore[current] + 1
                if 0 <= neighbor[0] < size and 0 <= neighbor[1] < size:
                    if maze[neighbor[1]][neighbor[0]] == 1:
                        continue
                else:
                    continue
                    
                if neighbor in close_set and tentative_g >= gscore.get(neighbor, 0):
                    continue
                    
                if tentative_g < gscore.get(neighbor, float('inf')) or neighbor not in [i[1] for i in oheap]:
                    came_from[neighbor] = current
                    gscore[neighbor] = tentative_g
                    fscore[neighbor] = tentative_g + self.heuristic(neighbor, goal)
                    heapq.heappush(oheap, (fscore[neighbor], neighbor))
        return []
    
    def escolher_alvo(self, world):            
        # Se estamos na fase de coleta e pegamos todos os pacotes, muda para fase de entrega
        if self.collection_phase and not world.packages and self.cargo > 0:
            print("Todos os pacotes coletados. Mudando para fase de entrega.")
            self.collection_phase = False
            self.need_replan = True
            return self.escolher_alvo(world)
        # Se precisamos replanejar ou não temos um caminho planejado
        elif self.need_replan or not self.full_path or self.current_target_index >= len(self.full_path):
            # Fase de coleta - pegar todos os pacotes
            if self.collection_phase and world.packages:
                print("Planejando rota de coleta...")
                self.full_path = self.find_optimal_route(world.packages.copy(), world)
                self.current_target_index = 0
                self.need_replan = False
                if not self.full_path:
                    return None
            # Fase de entrega - entregar todos os pacotes
            elif not self.collection_phase and world.goals and self.cargo > 0:
                print("Planejando rota de entrega...")
                self.full_path = self.find_optimal_route(world.goals.copy(), world)
                self.current_target_index = 0
                self.need_replan = False
                if not self.full_path:
                    return None
            else:
                return None

            
        # Retorna o próximo alvo no caminho planejado
        if self.current_target_index < len(self.full_path):
            target = self.full_path[self.current_target_index]
            self.current_target_index += 1
            return target
            
        return None
# ==========================
# CLASSE WORLD (MUNDO)
# ==========================
class World:
    def __init__(self, seed=None):
        if seed is not None:
            random.seed(seed)
        # Parâmetros do grid e janela
        self.maze_size = 30
        self.width = 600
        self.height = 600
        self.block_size = self.width // self.maze_size

        # Cria uma matriz 2D para planejamento de caminhos:
        # 0 = livre, 1 = obstáculo
        self.map = [[0 for _ in range(self.maze_size)] for _ in range(self.maze_size)]
        # Geração de obstáculos com padrão de linha (assembly line)
        self.generate_obstacles()
        # Gera a lista de paredes a partir da matriz
        self.walls = []
        for row in range(self.maze_size):
            for col in range(self.maze_size):
                if self.map[row][col] == 1:
                    self.walls.append((col, row))

        # Número total de itens (pacotes) a serem entregues
        self.total_items = 4

        # Geração dos locais de coleta (pacotes)
        self.packages = []
        # Aqui geramos 5 locais para coleta, garantindo uma opção extra
        while len(self.packages) < self.total_items + 1:
            x = random.randint(0, self.maze_size - 1)
            y = random.randint(0, self.maze_size - 1)
            if self.map[y][x] == 0 and [x, y] not in self.packages:
                self.packages.append([x, y])

        # Geração dos locais de entrega (metas)
        self.goals = []
        while len(self.goals) < self.total_items:
            x = random.randint(0, self.maze_size - 1)
            y = random.randint(0, self.maze_size - 1)
            if self.map[y][x] == 0 and [x, y] not in self.goals and [x, y] not in self.packages:
                self.goals.append([x, y])

        # Cria o jogador usando a classe DefaultPlayer (pode ser substituído por outra implementação)
        self.player = self.generate_player()

        # Coloca o recharger (recarga de bateria) próximo ao centro (região 3x3)
        self.recharger = self.generate_recharger()

        # Inicializa a janela do Pygame
        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Delivery Bot")

        # Carrega imagens para pacote, meta e recharger a partir de arquivos
        self.package_image = pygame.image.load("images/cargo.png")
        self.package_image = pygame.transform.scale(self.package_image, (self.block_size, self.block_size))

        self.goal_image = pygame.image.load("images/operator.png")
        self.goal_image = pygame.transform.scale(self.goal_image, (self.block_size, self.block_size))

        self.recharger_image = pygame.image.load("images/charging-station.png")
        self.recharger_image = pygame.transform.scale(self.recharger_image, (self.block_size, self.block_size))

        # Cores utilizadas para desenho (caso a imagem não seja usada)
        self.wall_color = (100, 100, 100)
        self.ground_color = (255, 255, 255)
        self.player_color = (0, 255, 0)
        self.path_color = (200, 200, 0)

    def generate_obstacles(self):
        """
        Gera obstáculos com sensação de linha de montagem:
         - Cria vários segmentos horizontais curtos com lacunas.
         - Cria vários segmentos verticais curtos com lacunas.
         - Cria um obstáculo em bloco grande (4x4 ou 6x6) simulando uma estrutura de suporte.
        """
        # Barragens horizontais curtas:
        for _ in range(7):
            row = random.randint(5, self.maze_size - 6)
            start = random.randint(0, self.maze_size - 10)
            length = random.randint(5, 10)
            for col in range(start, start + length):
                if random.random() < 0.7:
                    self.map[row][col] = 1

        # Barragens verticais curtas:
        for _ in range(7):
            col = random.randint(5, self.maze_size - 6)
            start = random.randint(0, self.maze_size - 10)
            length = random.randint(5, 10)
            for row in range(start, start + length):
                if random.random() < 0.7:
                    self.map[row][col] = 1

        # Obstáculo em bloco grande: bloco de tamanho 4x4 ou 6x6.
        block_size = random.choice([4, 6])
        max_row = self.maze_size - block_size
        max_col = self.maze_size - block_size
        top_row = random.randint(0, max_row)
        top_col = random.randint(0, max_col)
        for r in range(top_row, top_row + block_size):
            for c in range(top_col, top_col + block_size):
                self.map[r][c] = 1

    def generate_player(self):
        # Cria o jogador em uma célula livre que não seja de pacote ou meta.
        while True:
            x = random.randint(0, self.maze_size - 1)
            y = random.randint(0, self.maze_size - 1)
            if self.map[y][x] == 0 and [x, y] not in self.packages and [x, y] not in self.goals:
                return DefaultPlayer([x, y])

    def generate_recharger(self):
        # Coloca o recharger próximo ao centro
        center = self.maze_size // 2
        candidates = []

        for dx in (-1, 0, +1):
            for dy in (-1, 0, +1):
                x, y = center + dx, center + dy
                if (0 <= x < self.maze_size and 0 <= y < self.maze_size
                    and self.map[y][x] == 0
                    and [x, y] not in self.packages
                    and [x, y] not in self.goals
                    and [x, y ] != self.player.position):
                    candidates.append([x,y])


        if candidates:
            return random.choice(candidates)
        
        for y in range(self.maze_size):
            for x in range(self.maze_size):
                if(self.map[y][x] == 0
                    and [x, y] not in self.packages
                    and [x, y] not in self.goals
                    and [x, y ] != self.player.position):
                    return [x, y]
        return None

    def can_move_to(self, pos):
        x, y = pos
        if 0 <= x < self.maze_size and 0 <= y < self.maze_size:
            return self.map[y][x] == 0
        return False

    def draw_world(self, path=None):
        self.screen.fill(self.ground_color)
        # Desenha os obstáculos (paredes)
        for (x, y) in self.walls:
            rect = pygame.Rect(x * self.block_size, y * self.block_size, self.block_size, self.block_size)
            pygame.draw.rect(self.screen, self.wall_color, rect)
        # Desenha os locais de coleta (pacotes) utilizando a imagem
        for pkg in self.packages:
            x, y = pkg
            self.screen.blit(self.package_image, (x * self.block_size, y * self.block_size))
        # Desenha os locais de entrega (metas) utilizando a imagem
        for goal in self.goals:
            x, y = goal
            self.screen.blit(self.goal_image, (x * self.block_size, y * self.block_size))
        # Desenha o recharger utilizando a imagem
        if self.recharger:
            x, y = self.recharger
            self.screen.blit(self.recharger_image, (x * self.block_size, y * self.block_size))
        # Desenha o caminho, se fornecido
        if path:
            for pos in path:
                x, y = pos
                rect = pygame.Rect(x * self.block_size + self.block_size // 4,
                                   y * self.block_size + self.block_size // 4,
                                   self.block_size // 2, self.block_size // 2)
                pygame.draw.rect(self.screen, self.path_color, rect)
        # Desenha o jogador (retângulo colorido)
        x, y = self.player.position
        rect = pygame.Rect(x * self.block_size, y * self.block_size, self.block_size, self.block_size)
        pygame.draw.rect(self.screen, self.player_color, rect)
        pygame.display.flip()

# ==========================
# CLASSE MAZE: Lógica do jogo e planejamento de caminhos (A*)
# ==========================
class Maze:
    def __init__(self, seed=None):
        self.world = World(seed)
        self.running = True
        self.score = 0
        self.steps = 0
        self.delay = 50 # milissegundos entre movimentos
        self.path = []
        self.num_deliveries = 0  # contagem de entregas realizadas

    def heuristic(self, a, b):
        # Distância de Manhattan
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def astar(self, start, goal):
        maze = self.world.map
        size = self.world.maze_size
        neighbors = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        close_set = set()
        came_from = {}
        gscore = {tuple(start): 0}
        fscore = {tuple(start): self.heuristic(start, goal)}
        oheap = []
        heapq.heappush(oheap, (fscore[tuple(start)], tuple(start)))
        while oheap:
            current = heapq.heappop(oheap)[1]
            if list(current) == goal:
                data = []
                while current in came_from:
                    data.append(list(current))
                    current = came_from[current]
                data.reverse()
                return data
            close_set.add(current)
            for dx, dy in neighbors:
                neighbor = (current[0] + dx, current[1] + dy)
                tentative_g = gscore[current] + 1
                if 0 <= neighbor[0] < size and 0 <= neighbor[1] < size:
                    if maze[neighbor[1]][neighbor[0]] == 1:
                        continue
                else:
                    continue
                if neighbor in close_set and tentative_g >= gscore.get(neighbor, 0):
                    continue
                if tentative_g < gscore.get(neighbor, float('inf')) or neighbor not in [i[1] for i in oheap]:
                    came_from[neighbor] = current
                    gscore[neighbor] = tentative_g
                    fscore[neighbor] = tentative_g + self.heuristic(neighbor, goal)
                    heapq.heappush(oheap, (fscore[neighbor], neighbor))
        return []

    def game_loop(self):
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    return

            if self.num_deliveries >= self.world.total_items:
                self.running = False
                break

            # Verifica se há bateria suficiente para o próximo movimento
            if self.world.player.battery <= 0:
                if self.world.recharger:
                    path_to_recharger = self.astar(self.world.player.position, self.world.recharger)
                    if path_to_recharger:
                        print("Indo para o carregador - bateria crítica!")
                        for pos in path_to_recharger:
                            self.world.player.position = pos
                            self.steps += 1
                            self.world.player.battery -= 1
                            self.score -= 1 if self.world.player.battery >= 0 else 5
                            if pos == self.world.recharger:
                                self.world.player.battery = 60
                                print("Bateria recarregada!")
                                # Após recarregar, precisa replanejar
                                self.world.player.need_replan = True
                            self.world.draw_world(path_to_recharger)
                            pygame.time.wait(self.delay)
                        continue
                    else:
                        print("Não há caminho para o carregador!")
                        self.running = False
                        break
                else:
                    print("Sem recarregador disponível!")
                    self.running = False
                    break

            target = self.world.player.escolher_alvo(self.world)
            if target is None:
                if not self.world.packages and not self.world.goals and self.world.player.cargo == 0:
                    print("Missão completa! Todos os pacotes foram entregues.")
                else:
                    print("Sem alvo disponível mas missão não concluída.")
                self.running = False
                break

            # Move para o próximo alvo no caminho planejado
            next_pos = target
            if self.world.can_move_to(next_pos):
                self.world.player.position = next_pos
                self.steps += 1
                self.world.player.battery -= 1
                self.score -= 1 if self.world.player.battery >= 0 else 5
                
                # Verifica se chegou em um pacote ou meta
                if next_pos in self.world.packages and self.world.player.collection_phase:
                    self.world.player.cargo += 1
                    self.world.packages.remove(next_pos)
                    self.world.player.need_replan = True  # Precisa replanejar após pegar pacote
                    print("Pacote coletado em", next_pos, "Cargo agora:", self.world.player.cargo)
                elif next_pos in self.world.goals and not self.world.player.collection_phase and self.world.player.cargo > 0:
                    self.world.player.cargo -= 1
                    self.num_deliveries += 1
                    self.world.goals.remove(next_pos)
                    self.world.player.need_replan = True  # Precisa replanejar após entregar pacote
                    self.score += 50
                    print("Pacote entregue em", next_pos, "Cargo agora:", self.world.player.cargo)
                elif next_pos == self.world.recharger:
                    self.world.player.battery = 60
                    self.world.player.need_replan = True  # Precisa replanejar após recarregar
                    print("Bateria recarregada!")
                    
                self.world.draw_world([next_pos])
                pygame.time.wait(self.delay)

            print(f"Passos: {self.steps}, Pontuação: {self.score}, Cargo: {self.world.player.cargo}, Bateria: {self.world.player.battery}, Entregas: {self.num_deliveries}")

        print("Fim de jogo!")
        print("Pontuação final:", self.score)

# ==========================
# PONTO DE ENTRADA PRINCIPAL
# ==========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Delivery Bot: Navegue no grid, colete pacotes e realize entregas."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Valor do seed para recriar o mesmo mundo (opcional)."
    )
    args = parser.parse_args()
    
    maze = Maze(seed=args.seed)
    maze.game_loop()

