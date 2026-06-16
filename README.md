# Regresi-n-lineal-simple-MariaHerrero
Practica de regresión lineal simple del Modulo II-Bootcamp Factoría F5 P7


# Tarea: Investigación y Desarrollo de un Modelo de Regresión Lineal Simple

## Parte 1: Fundamentos de Regresión

---

### 1. ¿Qué es la regresión y cuál es su propósito en el Machine Learning? ¿En qué se diferencia de otros tipos de modelos supervisados como la clasificación?

Imagina que quieres adivinar cosas basándote en otras cosas que ya conoces. Por ejemplo:

- Si sabes cuántas horas estudiaste, ¿puedes adivinar qué nota sacarás?
- Si sabes cuántos años tiene un coche, ¿puedes adivinar cuánto cuesta?

**La regresión es eso: una herramienta que nos ayuda a PREDECIR un número basándonos en otros datos.**

**¿Cuál es su propósito?**

Es como tener una "fórmula mágica" que, después de ver muchos ejemplos, aprende a calcular un número concreto cuando le das información nueva.

**¿En qué se diferencia de la clasificación?**

| Regresión | Clasificación |
|-----------|---------------|
| Predice **números** (precios, temperaturas, notas) | Predice **categorías** (sí/no, perro/gato, rojo/verde/azul) |
| "¿Cuánto costará?" | "¿Es esto A o B?" |
| El resultado puede ser 15.3, 200, 0.8... | El resultado es una etiqueta: "spam", "no spam", "gato" |

**Ejemplo sencillo:**

- **Regresión**: "Con 5 años de antigüedad, el coche costará **7.500€**" (un número exacto)
- **Clasificación**: "Con estas características, el coche es **barato o caro**" (una categoría)

---

### 2. ¿Qué es la Regresión Lineal Simple y qué representa la ecuación ŷ = β₀ + β₁ · X? Explica qué significan β₀ (intercepto) y β₁ (pendiente) y cómo interpretarlos en un contexto real.

Imagina que dibujas una línea recta en un papel de cuadrícula. Esa línea te ayuda a adivinar un número a partir de otro número.

**La Regresión Lineal Simple** es la forma más básica de regresión: solo usas **UNA cosa** (una variable) para predecir **OTRA cosa**.

Por ejemplo:
- Usas las **horas de estudio** para predecir la **nota del examen**
- Usas los **metros cuadrados** de una casa para predecir su **precio**

**La ecuación: ŷ = β₀ + β₁ · X**

Vamos a desglosarla como si fuera una receta de cocina:

| Símbolo | Nombre sencillo | Qué significa |
|---------|----------------|---------------|
| **ŷ** (y con sombrerito) | "Mi predicción" | El número que el modelo calcula/adivina |
| **X** | "Lo que ya sé" | El dato que le doy al modelo (horas de estudio, metros de casa...) |
| **β₀** (beta cero) | **El intercepto** | Dónde "corta" la línea en el eje vertical |
| **β₁** (beta uno) | **La pendiente** | Cuánto sube o baja la línea cuando X aumenta |

**β₀ (Intercepto): El punto de partida**

Imagina que quieres calcular el precio de una pizza según sus ingredientes.
- **β₀** sería el **precio base de la pizza** (sin ingredientes, solo la masa y el queso)
- Es el valor de ŷ cuando X = 0

> **Ejemplo real**: Si estudias **0 horas** (X=0), ¿qué nota sacarías? Esa nota "base" sería β₀. (Aunque en la vida real, estudiar 0 horas no suele dar buena nota 😅)

**β₁ (Pendiente): La inclinación de la línea**

Es lo que le dice a la línea: **"por cada unidad que aumenta X, ¿cuánto cambia ŷ?"**

| Si β₁ es... | Significa que... |
|-------------|------------------|
| **Positivo** | Cuando X sube, ŷ también sube (más horas de estudio = mejor nota) |
| **Negativo** | Cuando X sube, ŷ baja (más años de coche = menos precio) |
| **Cero** | X no afecta a ŷ (la edad no influye en el precio de la pizza) |

> **Ejemplo real**: Si β₁ = 2, significa que **por cada hora extra de estudio**, tu nota sube **2 puntos**.

**Visualización mental:**

Imagina un gráfico de puntos (cada punto es un alumno: horas estudiadas vs nota obtenida).
La Regresión Lineal dibuja **la línea recta que mejor pasa por el medio de todos esos puntos**.

```
Nota
 10 |                    / (tu línea de predicción)
  8 |                 /
  6 |              /  ← β₁ dice cuánto inclinada está
  4 |           /
  2 |        /
  0 |_____/_____________
     0    2    4    6    8   (horas de estudio)
        ↑
       β₀ (aquí "corta" el eje vertical)
```

**Resumen con palabras sencillas:**
- **β₀** = "Con lo mínimo, ¿cuánto tengo?"
- **β₁** = "Por cada pasito que doy, ¿cuánto gano o pierdo?"

---

### 3. ¿Cuáles son los supuestos de la Regresión Lineal? Describe los supuestos principales: linealidad, homocedasticidad, independencia de errores y normalidad de residuos. ¿Por qué es importante verificarlos?

Imagina que quieres usar una regla para medir cosas. Pero la regla solo funciona bien si cumples ciertas condiciones: debe estar recta, debe estar en centímetros, etc.

La Regresión Lineal también tiene sus **"reglas del juego"** que deben cumplirse para que funcione bien. Son 4 supuestos principales:

**1. Linealidad**

**Qué significa:** La relación entre X e Y debe formar más o menos una **línea recta**.

**Ejemplo sencillo:** Si estudias más horas, tu nota sube de forma **proporcional y constante**. No es que estudiar 1 hora te dé +2 puntos, pero estudiar 2 horas te dé +10 puntos (eso sería una curva, no una línea).

```
✅ Bien:      ❌ Mal (no lineal):
Nota           Nota
 10 |    /      10 |      ___
  8 |  /         8 |    /
  6 |/           6 |  /
  4 |           4 | /
  2 |           2 |/
  0 |________   0 |________
     0  2  4       0  2  4
   (horas)       (horas)
```

**2. Homocedasticidad**

**Qué significa:** Los errores del modelo deben ser **parecidos** en todos los valores de X.

**En palabras sencillas:** El modelo no debe acertar muchísimo en unos casos y fallar muchísimo en otros. Los errores deben estar "repartidos uniformemente".

```
✅ Bien (puntos dispersos igual):     ❌ Mal (dispersión cambia):
    Y |  ·    ·                          Y |    ·
      | ·  ·   ·                         |  ··    ···
      |·    ·  ·                         | ·        ···
      |_________                         |___________
         X                                  X
      (siempre igual de "disperso")      (se abre como un abanico)
```

**Por qué importa:** Si los errores crecen, el modelo no es fiable para valores grandes de X.

**3. Independencia de errores**

**Qué significa:** Un error no debe "contagiar" al siguiente.

**Ejemplo sencillo:** Si predices el precio de casas y todas las casas de un mismo barrio están sobrevaloradas, los errores están **relacionados entre sí** (porque comparten barrio). Eso rompe el supuesto.

**Por qué importa:** Si los errores están "enredados", el modelo no aprende bien y las predicciones se vuelven inestables.

**4. Normalidad de residuos**

**Qué significa:** Los errores (residuos) deben formar una **campana de Gauss** — la mayoría cerca de cero, pocos muy lejos.

**En palabras sencillas:** El modelo debe fallar "poquito" la mayoría de las veces, y solo "mucho" en contadas ocasiones.

```
✅ Bien (campana):
      ∧
     / \     ← muchos errores pequeños
    /   \       cerca de 0
___/     \___
   -2  0  +2  (residuos)

❌ Mal (no campana):
      ___
     |   |    ← errores repartidos raro
_____|   |_____
```

**¿Por qué es IMPORTANTE verificarlos?**

| Si no cumples los supuestos... | Pasa esto... |
|-------------------------------|--------------|
| No linealidad | La línea recta no sirve, necesitarías una curva |
| No homocedasticidad | El modelo es muy bueno para unos datos y muy malo para otros |
| Errores dependientes | Las predicciones se "contagian" y no son fiables |
| No normalidad | Las métricas y confianza en el modelo son falsas |

**En resumen:** Es como intentar usar un martillo para atornillar. Si no cumples las condiciones, estás usando la herramienta equivocada para el problema.

---

### 4. ¿Qué es la función de coste (Loss Function) en regresión? Explica qué es el Error Cuadrático Medio (MSE) y el Error Absoluto Medio (MAE). ¿Cuál es la diferencia y cuándo preferirías usar uno u otro?

Imagina que estás jugando a los dardos 🎯. Cada vez que lanzas, puedes estar cerca o lejos del centro. La **función de coste** es como un **"medidor de qué tan malo fue tu lanzamiento"**.

En Machine Learning, la función de coste mide **qué tan lejos están las predicciones de mi modelo de los valores reales**. Es decir, **cuánto se equivoca el modelo**.

> **Objetivo:** Queremos que esta "puntuación de error" sea lo más **pequeña** posible.

**Error Cuadrático Medio (MSE)**

**Cómo funciona:** Toma cada error, lo **eleva al cuadrado**, y luego saca la media.

**Fórmula sencilla:**
```
MSE = promedio de (error × error)
```

**Ejemplo:**

| Real | Predicción | Error | Error al cuadrado |
|------|-----------|-------|-----------------|
| 10   | 8         | -2    | 4               |
| 5    | 7         | +2    | 4               |
| 8    | 8         | 0     | 0               |

**MSE = (4 + 4 + 0) ÷ 3 = 2.67**

**¿Qué hace especial al MSE?**
- **Castiga mucho los errores grandes** porque los eleva al cuadrado
- Si te equivocas por 10, el error es 100. Si te equivocas por 2, el error es 4.
- Es como un profesor estricto: **un error gordo te arruina la nota**

**Error Absoluto Medio (MAE)**

**Cómo funciona:** Toma cada error, le quita el signo (lo hace positivo), y saca la media.

**Fórmula sencilla:**
```
MAE = promedio de |error|
```

**Con el mismo ejemplo:**

| Real | Predicción | Error | Error absoluto |
|------|-----------|-------|---------------|
| 10   | 8         | -2    | 2             |
| 5    | 7         | +2    | 2             |
| 8    | 8         | 0     | 0             |

**MAE = (2 + 2 + 0) ÷ 3 = 1.33**

**¿Qué hace especial al MAE?**
- **Trata todos los errores igual** — no castiga más los errores grandes
- Es como un profesor justo: **un error de 10 puntos cuenta como 10, no como 100**

**¿Cuál es la diferencia?**

| | MSE | MAE |
|--|-----|-----|
| **Cálculo** | Error² | \|Error\| |
| **Trata errores grandes** | Muy mal (los castiga) | Normal |
| **Es más sensible a** | Valores extremos (outliers) | Todos los errores por igual |
| **Resultado** | Siempre más alto que MAE | Más bajo y directo |

**¿Cuándo usar uno u otro?**

| Usa **MSE** cuando... | Usa **MAE** cuando... |
|------------------------|----------------------|
| Los errores grandes son **muy malos** (ej: predicción de temperaturas donde un error de 20° es peligroso) | Quieres que todos los errores cuenten **igual** |
| No tienes valores extremos raros (outliers) | Tienes datos con **valores raros** que no quieres que dominen |
| Quieres que el modelo "se esfuerce" más en acertar todo | Quieres una medida más **realista** del error típico |

**Ejemplo cotidiano:**
- **MSE** = Notas de examen donde un 0 es catastrófico y quieres evitarlo a toda costa
- **MAE** = Estimar cuánto tardas en llegar al cole — un día te atrasa mucho y otro poco, pero quieres la media realista

---

### 5. ¿Qué significa dividir los datos en train y test? ¿Por qué es necesario? ¿Qué problema evitamos al no evaluar el modelo con los mismos datos con los que lo entrenamos?

Imagina que estudias para un examen de matemáticas 📚. Tienes un libro con 100 problemas.

**Opción A (la mala):** Estudias los 100 problemas y luego el examen te pone... ¡los mismos 100 problemas! Pues claro que sacas un 10, pero ¿realmente sabes matemáticas o te los aprendiste de memoria?

**Opción B (la buena):** Estudias 80 problemas del libro, y los otros 20 los guardas para hacer un **examen de prueba** antes del examen real. Así compruebas si de verdad entendiste la materia.

**Dividir en train y test es exactamente eso:**

| Parte | Nombre | Para qué sirve |
|-------|--------|----------------|
| **Train** (entrenamiento) | "Los problemas que estudio" | El modelo aprende, encuentra patrones, ajusta su fórmula |
| **Test** (prueba) | "El examen sorpresa" | Comprobar si el modelo aprendió de verdad o solo se memorizó |

**¿Por qué es necesario?**

Porque necesitamos saber si nuestro modelo es **útil para datos NUEVOS** que nunca ha visto.

> **En la vida real**, el modelo no va a predecir cosas que ya conoce. Va a predecir cosas **futuras** o **desconocidas**.

**¿Qué problema evitamos?**

El problema se llama **"sobreajuste" o overfitting** (ya lo veremos en la pregunta 8, pero te adelanto la idea):

Si evaluamos con los mismos datos de entrenamiento...

```
❌ MAL:
Datos de entrenamiento → Modelo → Evaluamos con los MISMOS datos
Resultado: "¡Soy un genio! ¡99% de acierto!"

Pero luego en la vida real:
Datos nuevos → Modelo → "¡Ay, no sé qué hacer! 😱"
```

Es como si te preguntaran el examen A pero tú solo habías estudiado el examen A de memoria. Cuando te dan el examen B, fracasas.

**¿Cómo se hace normalmente?**

- **80% train / 20% test** (lo más común)
- **70% train / 30% test** (si tienes pocos datos)

Y es importante que sea **aleatorio** — no cojas los primeros 80 y los últimos 20, porque podría haber algún patrón raro (por ejemplo, los datos están ordenados por fecha).

**Resumen:**

| | Train | Test |
|--|-------|------|
| 🎯 | Aprender | Evaluar |
| 📖 | Libro de estudio | Examen sorpresa |
| 🧠 | El modelo se entrena | Comprobamos si entendió |
| ⚠️ | NO evaluamos aquí | SÍ evaluamos aquí |

---

### 6. ¿Qué métricas se usan para evaluar un modelo de regresión?

Imagina que acabas de hacer un examen y quieres saber **qué tan bien te fue**. Puedes mirarlo de muchas formas: la nota final, cuántos fallaste, cuánto te diste de la nota máxima...

Las métricas de regresión son eso: **distintas formas de "calificar" a tu modelo**.

**R² (Coeficiente de Determinación)**

**Qué mide:** Qué tan bien tu modelo explica la realidad. Va de **0 a 1** (a veces puede ser negativo, pero eso es malísimo).

**Fórmula:**
```
R² = 1 - (suma de errores² del modelo / suma de errores² de la media)
```

**En palabras sencillas:**

Imagina que hay una persona muy vaga que, para cualquier pregunta, siempre responde **la media** (el promedio de todos los datos). R² compara tu modelo contra esa persona vaga.

| R² = 1 | R² = 0 | R² negativo |
|--------|--------|-------------|
| ¡Perfecto! Tu modelo acierta todo | Tu modelo es igual de malo que decir siempre la media | ¡Peor que decir la media! |

**Ejemplo:** Si R² = 0.85, significa que tu modelo explica el **85%** de lo que pasa. El 15% restante es "ruido" o cosas que no puedes predecir.

**MSE (Mean Squared Error)**

**Qué mide:** El error cuadrático medio. Ya lo vimos en la pregunta 4.

**Fórmula:**
```
MSE = (1/n) × Σ(yᵢ - ŷᵢ)²
```

**En palabras sencillas:** Suma todos los errores al cuadrado y divide por cuántos hay.

> **Nota:** Cuanto más **bajo**, mejor.

**RMSE (Root Mean Squared Error)**

**Qué mide:** Es la **raíz cuadrada** del MSE.

**Fórmula:**
```
RMSE = √MSE = √[(1/n) × Σ(yᵢ - ŷᵢ)²]
```

**¿Por qué existe?** Porque el MSE está en unidades "al cuadrado" (si predices euros, el MSE está en "euros cuadrados" — ¡qué raro!). El RMSE vuelve a las **unidades originales**.

**Ejemplo:** Si predices precios de casas en euros:
- MSE = 4.000.000 €² (¿qué es eso?)
- RMSE = 2.000 € (¡ahora sí tiene sentido!)

> **Cuanto más bajo, mejor.**

**MAE (Mean Absolute Error)**

**Qué mide:** El error absoluto medio. También lo vimos en la pregunta 4.

**Fórmula:**
```
MAE = (1/n) × Σ|yᵢ - ŷᵢ|
```

**En palabras sencillas:** Suma todos los errores (quitando los signos negativos) y divide por cuántos hay.

> **Cuanto más bajo, mejor.**

**Resumen comparativo:**

| Métrica | ¿Qué dice? | ¿Cuándo usarla? |
|---------|-----------|-----------------|
| **R²** | "¿Qué tan bueno soy comparado con decir la media?" | Para saber si tu modelo vale la pena |
| **MSE** | "Error cuadrático medio" | Cuando quieres castigar errores grandes |
| **RMSE** | "Error típico en las unidades originales" | Para interpretar fácilmente el error |
| **MAE** | "Error medio sin cuadrados" | Para una medida justa y realista |

**Ejemplo práctico con números:**

Imagina que predices notas de examen:

| Real (y) | Predicción (ŷ) | Error |
|----------|---------------|-------|
| 8 | 6 | -2 |
| 5 | 7 | +2 |
| 10 | 9 | -1 |

- **MAE** = (2 + 2 + 1) / 3 = **1.67** → "Me equivoco de media en 1.67 puntos"
- **MSE** = (4 + 4 + 1) / 3 = **3.0**
- **RMSE** = √3.0 = **1.73** → "Mi error típico es de 1.73 puntos"
- **R²** = depende de la media de los datos reales

---

### 7. ¿Qué son los residuos y para qué sirve analizarlos?

Imagina que pides una pizza 🍕 y te dicen que llegará en 30 minutos. Llega en 35 minutos.

- **Tu predicción:** 30 minutos
- **La realidad:** 35 minutos
- **El residuo:** 35 - 30 = **+5 minutos** (te fallaron por 5 minutos)

**Los residuos son los errores del modelo:** la diferencia entre lo que predijo y lo que realmente pasó.

```
Residuo = Valor real - Valor predicho
```

**¿Qué indica un residuo cercano a 0?**

Si el residuo es **0** → ¡El modelo acertó a la perfección!

Si el residuo es **cercano a 0** → El modelo se equivocó poquito, está bien.

**Ejemplo:**

| Real | Predicción | Residuo | Significado |
|------|-----------|---------|-------------|
| 10 | 10.2 | -0.2 | ¡Casi perfecto! |
| 8 | 3 | +5 | Falló por mucho |
| 5 | 5 | 0 | ¡Acierto total! |

**¿Para qué sirve analizarlos?**

Los residuos son como los **"mensajes secretos"** que el modelo te envía sobre cómo está funcionando. Si los miras bien, descubres problemas.

**¿Qué patrón en los residuos revelaría que el modelo no es adecuado?**

Imagina que dibujas un gráfico de residuos (eje X = predicciones, eje Y = residuos):

**Buen modelo (residuos aleatorios):**
```
Residuo
  +3 |    ·  ·
  +1 |  ·    · ·
   0 |___________
  -1 | ·  ·    ·
  -3 |   ·  ·
        (predicciones)
```
**Forma:** Puntos dispersos como nube, **sin forma**. Es como lanzar confeti al aire — sin patrón.

**Mal modelo (patrones raros):**

**Patrón 1: Forma de U o arco** → ¡No es lineal! Necesitas una curva.
```
Residuo
  +3 |  ·    ·
   0 |_______/ \______
  -3 |      ·   ·
```
**Significado:** El modelo se equivoca sistemáticamente en los extremos. La relación real es una curva, no una línea.

**Patrón 2: Abanico (se abre o se cierra)** → ¡No hay homocedasticidad!
```
Residuo
  +3 |        ·  ·
  +1 |    · ·
   0 |___________
  -1 | · ·
  -3 |·
```
**Significado:** El modelo acierta bien para valores pequeños, pero falla mucho para valores grandes (o viceversa).

**Patrón 3: Línea diagonal** → ¡Algo está mal en el modelo!
```
Residuo
  +3 |              /
  +1 |           /
   0 |________/
  -3 |
```
**Significado:** Hay una relación que el modelo no está capturando. Quizás falta una variable importante.

**Resumen visual:**

| Forma de los residuos | ¿Qué significa? |
|------------------------|-----------------|
| Nube sin forma (aleatoria) | ¡Modelo bien! |
| U, arco, o curva | Necesitas una relación no lineal |
| Abanico (se abre/se cierra) | Los errores no son estables |
| Línea o patrón claro | Falta algo en el modelo |

**En una frase:**

> **Los residuos son como el "diagnóstico médico" del modelo.** Si tienen patrón, el modelo está enfermo. Si son aleatorios, está sano.

---

### 8. ¿Qué es el overfitting y cómo se detecta comparando R² en train vs. test?

Imagina que estudias para un examen de historia 📚...

**Caso A:** Estudias los temas, entiendes las causas de las guerras, las fechas importantes... Llega el examen y sacas un 8. ¡Bien! Sabes historia.

**Caso B:** Te memorizas **palabra por palabra** las 50 preguntas del libro de ejercicios. No entiendes nada, pero te sabes las respuestas de memoria. El examen te pone... ¡las mismas 50 preguntas! Sacas un 10. Pero si te preguntan algo ligeramente diferente, no sabes qué decir.

**El Caso B es overfitting (sobreajuste):** El modelo se aprende los datos de memoria en vez de aprender la lógica real.

**¿Qué es el overfitting?**

Es cuando el modelo es **demasiado complejo** para los datos y, en vez de encontrar patrones generales, **memoriza los ejemplos** que ha visto.

| | Buen modelo | Overfitting |
|--|-------------|-------------|
| En train | R² alto (bien) | R² muy alto (casi perfecto) |
| En test | R² alto también | R² bajo o muy bajo |
| En la vida real | Funciona bien | Fracasa estrepitosamente |

**¿Cómo se detecta comparando R²?**

Es muy sencillo. Miras el R² en **train** y en **test**:

**Modelo equilibrado (bien):**
```
R² en train:  0.85  
R² en test:   0.82  
              ¡Muy parecidos! El modelo aprendió de verdad.
```

**Overfitting (mal):**
```
R² en train:  0.99  
R² en test:   0.45  
              ¡Abismo! Memorizó train, pero en test fracasa.
```

**La regla de oro:**

| Diferencia train vs. test | Significado |
|---------------------------|-------------|
| R² train ≈ R² test (parecidos) | Modelo sano |
| R² train >> R² test (muy diferentes) | Overfitting |
| Ambos bajos | Underfitting (modelo muy simple) |

> **">>"** significa "mucho mayor que"

**Ejemplo con números concretos:**

Imagina un modelo que predice precios de casas:

| | R² | Interpretación |
|--|-----|----------------|
| **Train** | 0.98 | "¡Soy un genio!" |
| **Test** | 0.52 | "Ay, no entiendo nada..." |

**Diagnóstico:** ¡Overfitting! El modelo se aprendió los precios de memoria en vez de entender qué hace que una casa valga más.

**¿Por qué pasa el overfitting?**

- **Modelo muy complejo** (muchos parámetros para pocos datos)
- **Pocos datos** (es fácil memorizar 10 ejemplos, imposible memorizar 1 millón)
- **Entrenar demasiado tiempo** (el modelo "fuerza" a encajar todo)

**¿Cómo se arregla?**

| Solución | Qué hace |
|----------|----------|
| Más datos | Más difícil de memorizar |
| Modelo más simple | Menos "capacidad de memorización" |
| Regularización | Castiga al modelo si se vuelve demasiado complejo |
| Validación cruzada | La siguiente pregunta |

**Resumen:**

| | Train | Test | Diagnóstico |
|--|-------|------|-------------|
| Bien | 0.85 | 0.83 | ¡Perfecto! |
| Dudoso | 0.60 | 0.58 | Underfitting (muy simple) |
| Mal | 0.99 | 0.40 | **Overfitting** (memorizó) |

---

### 9. ¿Qué es la validación cruzada (cross-validation) y qué ventaja ofrece frente a una sola división train/test?

Imagina que vas a cocinar una receta nueva 🍰. Quieres saber si queda buena, así que la pruebas. Pero en vez de probar solo un cachito...

**Opción A (train/test simple):** Cortas la tarta en 2 partes: 80% para hacerla, 20% para probarla. Pruebas solo ese 20%. ¿Y si justo ese pedazo tenía mucha nuez y el resto no? Tu prueba no es muy fiable.

**Opción B (cross-validation):** Cortas la tarta en 5 pedazos. Pruebas 5 veces, cada vez usando un pedazo diferente para probar y los otros 4 para hacerla. Al final, haces la media de las 5 pruebas.

**¡La Opción B es cross-validation!** Pruebas tu modelo de varias formas para estar más seguro.

**¿Cómo funciona paso a paso?**

Imagina que tienes 100 datos y usas **K-Fold Cross-Validation** (la más común), con K=5 (5 "pedazos"):

```
Datos: [1][2][3][4][5][6][7][8][9][10]... (100 datos)

Vuelta 1:  Train [2][3][4][5]  →  Test [1]  →  R² = 0.82
Vuelta 2:  Train [1][3][4][5]  →  Test [2]  →  R² = 0.85
Vuelta 3:  Train [1][2][4][5]  →  Test [3]  →  R² = 0.80
Vuelta 4:  Train [1][2][3][5]  →  Test [4]  →  R² = 0.83
Vuelta 5:  Train [1][2][3][4]  →  Test [5]  →  R² = 0.84

Resultado final: Media de todos los R² = 0.828
```

**En cada vuelta**, todos los datos son **train** alguna vez y **test** alguna vez.

**¿Qué ventaja tiene frente a una sola división train/test?**

| | Una sola división train/test | Cross-validation |
|--|------------------------------|------------------|
| **Suerte** | Si te tocan datos raros en test, el resultado miente | Pruebas con TODOS los datos, la suerte se equilibra |
| **Confianza** | Un solo número: ¿es bueno o tuve suerte? | Media de varios números: más fiable |
| **Datos** | Desperdicias datos en test que no usas para entrenar | Cada dato sirve tanto para train como para test |
| **Robustez** | Resultado puede variar mucho si cambias la división | Resultado más estable y repetible |

**Analogía:**

| | Train/test simple | Cross-validation |
|--|-------------------|------------------|
| Es como... | Hacer **un examen** y quedarte con esa nota | Hacer **5 exámenes** y quedarte con la media |

¿En cuál confías más para saber si realmente dominas la materia? ¡En la media de 5!

**¿Cuándo usar cada uno?**

| Usa **train/test simple** cuando... | Usa **cross-validation** cuando... |
-------------------------------------|-----------------------------------|
| Tienes **muchos datos** (millones) | Tienes **pocos datos** |
| Quieres **rapidez** (es más rápido) | Quieres **máxima confianza** en el resultado |
| Ya estás seguro de tu modelo | Estás **comparando muchos modelos** |
| Estás en producción ya final | Estás en la fase de **investigación/experimento** |

**Resumen:**

| | Ventaja |
|--|---------|
| Usa todos los datos para entrenar Y para test |
| Resultado más fiable y estable |
| Evita que la "suerte" de una división te engañe |
| Mejor para comparar modelos entre sí |

---

## Parte 2: Notebook de Implementación en Python

El notebook completo de implementación se encuentra en el archivo `proyecto_regresion_lineal.ipynb`.

### Resumen de resultados obtenidos:

- **Dataset:** Salary Dataset (30 muestras, 2 variables)
- **Variable X:** Años de Experiencia
- **Variable y:** Salario
- **Ecuación del modelo:** Salario = 25202.89 + 9731.20 × Años de Experiencia
- **R² en test:** 0.90 (excelente)
- **MAE:** ~$6,871
- **RMSE:** ~$7,302
- **Validación cruzada (5-Fold):** R² promedio = 0.97 (muy estable)
- **Overfitting:** No detectado (diferencia train-test moderada)
- **Outliers:** Ninguno detectado

### Conceptos aplicados:

| Concepto | Descripción |
|---|---|
| **Regresión Lineal** | Ajusta una línea recta entre X e y |
| **β₀ y β₁** | Intercepto y pendiente de la recta |
| **Train/Test Split** | División de datos para evaluación honesta |
| **R²** | Porcentaje de varianza explicada (0 a 1) |
| **MAE / RMSE** | Magnitud del error en unidades originales |
| **Análisis de residuos** | Diagnóstico de supuestos del modelo |
| **Validación cruzada** | Evaluación robusta con múltiples splits |

---

*Entrega realizada para el Bootcamp de Machine Learning. Maria Herrero*
