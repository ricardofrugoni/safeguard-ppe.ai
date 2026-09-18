# Imagens para a documentação

Este diretório reúne as imagens exibidas na galeria do [README principal](../../README.md). Os arquivos SVG atuais são espaços reservados, sem resultados ou amostras reais.

## Como preencher a galeria

1. Adicione uma imagem JPG, JPEG, PNG ou WebP neste diretório.
2. No README principal, substitua o caminho do marcador `.svg` pelo arquivo real, incluindo a extensão correta.
3. Atualize a legenda com a origem, o conjunto de dados e o identificador do experimento. Para predições, informe o arquivo de pesos e o limiar de confiança.
4. Confira a prévia do Markdown para verificar legibilidade e enquadramento.

| Espaço | Nome sugerido para a imagem real | Conteúdo esperado |
| --- | --- | --- |
| Imagem do dataset | `dataset-original.jpg` | Cena original pertencente ao conjunto de treino. |
| Anotações de referência | `dataset-anotado.jpg` | A mesma cena com as caixas e classes das anotações, sem confundi-las com predições. |
| Lote de treinamento | `lote-treinamento.jpg` | Visualização de um lote real, incluindo transformações aplicadas no treinamento. |
| Predição em validação | `resultado-inferencia.jpg` | Saída do modelo em uma imagem identificada como pertencente à validação. |

Exemplo de legenda:

> Hard Hat Workers v2 · validação · experimento `<identificador>` · pesos `<arquivo>` · confiança mínima `<valor>`.

Para gerar gráficos e visualizações automáticas do treinamento, habilite `plots=True` na chamada de `model.train(...)`. O código atual utiliza `plots=False`. Identifique separadamente as curvas de treinamento, as anotações de referência e as predições ao adicioná-las à documentação.

Inclua apenas amostras que possam ser publicadas e registre a atribuição da fonte quando aplicável. Use imagens legíveis, preferencialmente com até 1.200 pixels de largura, sem inserir o dataset completo nesta pasta. Exemplos de falsos positivos e falsos negativos também ajudam a documentar os limites do modelo.

As exceções em `.gitignore` permitem versionar essas imagens; os datasets e os pesos de treinamento continuam fora do Git.
