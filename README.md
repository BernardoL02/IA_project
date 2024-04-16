# IA_project


                  Inteligência Artificial
                  2023/2024 – 2º Semestre




              PROJETO – CLASSIFICAÇÃO DE IMAGENS




1. Objetivos
Neste trabalho os alunos deverão:
     ▪   Utilizar um conjunto de dados de imagens pensado para uma tarefa de classificação;
     ▪   Treinar modelos baseados em redes neuronais convolucionais;
     ▪   Escrever um relatório usando anotações markdown nos notebooks desenvolvidos. O
         relatório deve conter:
             o Todas as etapas realizadas para construção dos modelos;
             o A descrição de todas as experiências e resultados obtidos. A análise dos
                  resultados deve incluir a análise de métricas (por exemplo, matrizes de confusão,
                  accuracy, precision, recall e F1 score), gráficos adequados e a análise dos
                  resultados.
----------------------------------------------------------------------------------------------------
2. Conjunto de dados
  O conjunto de dados disponibilizado no Moodle está dividido em 6 diretorias: 5 diretorias train#
  e a diretoria test. Todos os grupos utilizam a mesma diretoria de teste. Cada grupo deve usar 4
  das diretorias train# como conjunto de treino e a diretoria train# restante como conjunto de
  validação. A diretoria a ser utilizada por cada grupo como conjunto de validação é determinada
  da seguinte forma: calcular a soma do último dígito do número de estudante de cada elemento
  do grupo e depois fazer o resto da divisão por 5 e, finalmente, somar 1. Por exemplo:
  Número do estudante 1: 2200783
  Número do estudante 2: 2243929
  12 % 5 + 1 = 2 + 1 = 3
  Conjunto de validação: train3
  Conjunto de treino composto pelas imagens das diretorias train1, train2, train4 e train5

----------------------------------------------------------------------------------------------------
3. Requisitos
  O projeto possui os seguintes requisitos:
      ▪    Devem ser utilizados e descritos conjuntos de dados de treino, validação e teste;
      ▪    Devem ser utilizadas imagens RGB (três canais);
      ▪    Deve ser desenvolvido pelo menos um modelo de raiz (que aqui denominamos por
           modelo S). Pelo menos um destes modelos deve ser diferente do modelo desenvolvido
           nas aulas;
      ▪    Devem ser explorados pelo menos dois otimizadores distintos;
      ▪    Os modelos S devem ser treinados com e sem data augmentation;
      ▪    Devem ser desenvolvidos modelos utilizando transfer learning (que aqui denominamos
           por modelos T) usando as técnicas de feature extraction e fine tuning;
      ▪    Os modelos T também devem ser treinados com e sem data augmentation.


----------------------------------------------------------------------------------------------------
4. Cotações
    05% - Processamento dos dados
    35% - Modelos S
    30% - Modelos T
    20% - Relatório
    10% - Extras
A avaliação do projeto favorecerá a capacidade de inovação dos estudantes, ou seja, de irem
além dos conteúdos aprendidos nas aulas (fichas e hands-ons).
    Exemplos de extras:
     • Utilizar técnicas de regularização;
     •    Deployment dos modelos desenvolvidos numa aplicação (standalone ou web);
     •    Desenvolvimento de operações de data augmentation customizadas e adequadas ao
          problema.

----------------------------------------------------------------------------------------------------
5. Prazos, datas, regras e instruções
  5.1. Data limite de entrega do projeto: 22 de junho de 2024, 23:59.
  5.2. O projeto é realizado em grupos de 2 estudantes. Não são aceites projetos realizados por
     grupos com mais de 2 elementos. Os estudantes que pretendam realizar o projeto
     individualmente devem solicitá-lo, por escrito, ao docente responsável pela UC. Apenas em
     casos bem fundamentados serão autorizados projetos realizados individualmente.
  5.3. O projeto deve ser entregue em arquivo zip seguindo o formato dl_project_#1_#2.zip, onde
     #1 e #2 devem ser substituídos pelos números dos alunos dos elementos do grupo. O arquivo
     zip deve conter:
     •    os notebooks (ficheiros .ipynb) completos após a execução;
     •    os notebooks (ficheiros .ipynb) sem conteúdo markdown e antes de serem executados;
     •    pdfs com os notebooks com os resultados da execução;
     •    os modelos;
     •    as features calculadas (para modelos desenvolvidos usando transfer learning sem data
          augmentation).
  
  5.4. Poderá ser realizada uma prova oral em casos em que os docentes considerem necessário. A
     nota da prova oral (de 0 a 100%) multiplica pela nota do projeto. A lista de estudantes a
     realizar prova oral será publicada no Moodle depois de realizada a entrega.




                                                                                    

