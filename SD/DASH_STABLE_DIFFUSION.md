# DASH w Stable Diffusion

Ten dokument opisuje obecne użycie DASH w części `SD/` repozytorium. Ma służyć jako punkt startowy do zrozumienia, jak DASH działa ze Stable Diffusion, jak interpretować targety attention/resnet oraz jak sensownie podejść do rozszerzenia tego mechanizmu na NSFW.

## Krótka definicja

DASH jest obecnie używany jako warm-start przed właściwym treningiem unlearningu. Nie jest osobną pętlą optymalizacji. Pipeline:

1. ładuje bazowy Stable Diffusion,
2. przygotowuje retain/forget data,
3. wykonuje `run_dash_sd_warm_start(...)`,
4. bezpośrednio skaluje wybrane wagi U-Netu przez `shrink`,
5. dopiero potem uruchamia główny trening unlearningu (`roft`, `rl`, `ga`, `intact`).

W class-forgetting DASH jest wykonywany w `random_label.py`, `gradient_ascent.py` i classowym `intact_unlearn.py`. Konfiguracja przechodzi z `pipeline.py` przez `dash_config`.

## Główne pliki

- `SD/train-scripts/dash_sd_runtime.py` - runtime DASH dla Stable Diffusion.
- `SD/train-scripts/dash_sd_targets.py` - wybór wag U-Netu, które mogą być modyfikowane.
- `DASH/plasticity_common.py` - wspólna logika `global` / `per_filter`.
- `SD/configs/pipeline_class.yaml` - aktualny config class-forgetting.
- `SD/pipeline.py` - przekazywanie `dash_config` do class-forgetting.

## Konfiguracja

Najważniejsza sekcja:

```yaml
dash:
  warm_start: false
  target: "unet_resnet"
  signal_mode: "retain_only"
  plasticity_granularity: "per_filter"
  grad_aggregation: "ema"
  alpha: 0.1
  num_aug: 1
  min_shrink: 0.5
  svd_truncate_evr: 0.95
  preserve_forget_evr: 0.95
  include_bias: false
  log_cosine_histograms: true
  cosine_hist_bins: 50
  retain_batches: 8
  forget_batches: 8
```

`warm_start: true` włącza DASH. Jeśli jest `false`, runtime zwraca statystyki z `dash_sd_enabled=0.0` i nie modyfikuje wag.

## Targety U-Netu

DASH wybiera tylko moduły `Linear` i `Conv2d` z `model.model.diffusion_model`. Domyślnie bierze tylko `weight`; biasy są pomijane, dopóki `include_bias: false`.

Targety:

- `unet` / `unet_all` / `all`
  - wszystkie wagi `Linear` i `Conv2d` w U-Necie,
  - obejmuje attention, resblocki i pozostałe projekcje/konwolucje.

- `unet_xattn`
  - tylko moduły, których ścieżka zawiera `attn2`,
  - w Stable Diffusion oznacza to cross-attention.

- `unet_attn`
  - wszystkie moduły attention,
  - warunek jest szerszy niż `unet_xattn`: ścieżka zawiera `attn` albo typ modułu zawiera `attention`,
  - obejmuje self-attention i cross-attention.

- `unet_resnet` / `unet_resblock`
  - moduły w ścieżkach ResBlock/ResNet,
  - praktycznie residual/convolutional backbone U-Netu.

Najmniejszy obecny target obejmujący jednocześnie `unet_attn` i `unet_resnet` to `unet`, ale `unet` obejmuje też dodatkowe `Linear`/`Conv2d`, nie tylko sumę attention+resnet.

## Biasy

Obecnie DASH w class configu nie modyfikuje biasów:

```yaml
dash:
  include_bias: false
```

W kodzie bias jest dodawany tylko, jeśli `include_bias=true` i moduł ma `module.bias`.

Rekomendacja na teraz: nie włączać biasów jako pierwszego rozszerzenia. Biasy mają mało parametrów, ale mogą przesuwać aktywacje globalnie. Najpierw warto zrozumieć zachowanie wag `Linear`/`Conv2d` przez histogramy alignmentu i shrinku.

## Granularity

`plasticity_granularity` kontroluje, jak DASH dzieli tensor na jednostki decyzyjne.

- `global`
  - cały tensor jest jedną jednostką,
  - jeden cosine/shrink dla całej wagi.

- `per_filter`
  - dla `Conv2d [Cout, Cin, kh, kw]`: jednostka = jeden output filter / output channel,
  - dla `Linear [out_features, in_features]`: jednostka = jeden output row,
  - dla biasu `[out_features]`, gdyby bias był włączony: jednostka = pojedynczy scalar.

Obecny config używa:

```yaml
plasticity_granularity: "per_filter"
```

To jest naturalne dla konwolucji. Dla attention działa jako row-wise na macierzach `to_q`, `to_k`, `to_v`, `to_out`, ale nie jest head-aware.

## Attention: obecne ograniczenie

DASH nie wie obecnie, które wiersze `Linear` należą do którego attention head. Dla attention `per_filter` oznacza “per output row”, a nie “per head”.

To jest sensowny baseline, ale semantycznie słabszy niż wariant head-aware:

- row-wise może shrinkować różne wiersze tego samego heada inaczej,
- head-aware traktowałby cały head jako jednostkę,
- head-aware byłby naturalniejszy dla pytań typu “które heady są odpowiedzialne za zapominanie/retencję?”.

Rekomendowana kolejność eksperymentów dla attention:

1. `target: unet_xattn`
   - najwęższy wariant cross-attention,
   - dobry, gdy zapominanie jest silnie związane z promptem/koncepcją tekstową.

2. `target: unet_attn`
   - szerszy attention: self+cross,
   - dobry test, czy self-attention też przenosi sygnał zapominania.

3. `target: unet`
   - obejmuje attention i resnet,
   - najmniejszy obecny wariant, który obejmuje jednocześnie attention i resblocki.

4. Dodać nowy tryb `attention_head`
   - przyszłe rozszerzenie,
   - wymaga mapowania `Linear` attention na głowy,
   - przydatne szczególnie dla `unet_xattn` i `unet_attn`.

## Signal modes

`signal_mode` określa, z jakich gradientów DASH buduje kierunek shrinku.

- `retain_only`
  - używa retain gradient,
  - najprostszy i obecnie najczęściej używany wariant w sweepach,
  - nie wymaga aktywnego sygnału forget do decyzji.

- `forget_perp_retain`
  - używa forget gradientu po odjęciu komponentu zgodnego z retain,
  - wymaga forget loadera,
  - próbuje oddzielić forget od retain.

- `preserve_complement`
  - buduje kierunek preserve, a mocniej shrinkuje complement,
  - używa `preserve_forget_evr` i projekcji,
  - bardziej złożony i bardziej wrażliwy na jakość estymacji gradientów.

## Alignment, shrink i logowanie

Alignment to wartości cosinusa między aktualną wagą a kierunkiem wynikającym z gradientu DASH. Wartości są w `[-1, 1]`.

Interpretacja:

- cosine dodatni: kierunek jest bardziej zgodny,
- cosine bliski zeru: słabe lub ortogonalne dopasowanie,
- cosine ujemny: konflikt kierunków.

Obecne logowanie obejmuje:

- globalne statystyki alignmentu i shrinku,
- globalny histogram cosine,
- globalną CDF cosine,
- per-moduł histogram cosine,
- per-moduł CDF cosine,
- per-moduł medianę cosine liczona z surowych scalarów alignmentu, nie ze środka bina histogramu.

Cel per-moduł logowania: zobaczyć, czy np. `attn2.to_q`, `attn2.to_k`, `attn2.to_v`, `to_out` albo resblocki mają różne rozkłady alignmentu. Globalny histogram może ukrywać takie różnice.

`negative_fraction`, `near_zero_fraction`, `positive_fraction` są obecnie logowane globalnie:

- `negative_fraction`: masa histogramu z cosine < -0.05,
- `near_zero_fraction`: masa w [-0.05, 0.05],
- `positive_fraction`: masa z cosine > 0.05.

Per moduł nie logujemy tych frakcji, żeby nie produkować zbyt wielu scalarów.

## Jak czytać histogram i CDF

Histogram mówi, gdzie leży masa alignmentu. CDF plot mówi, jaki procent jednostek ma alignment mniejszy lub równy danemu progowi. Scalarowe `cdf/bin_XX` nie są logowane, żeby nie produkować dziesiątek paneli W&B. Mediana jest osobnym scalarem liczonym bezpośrednio z raw alignment values, więc nie jest kwantyzowana do środka bina histogramu.

Przykłady pytań:

- Czy większość `attn2.to_v` ma cosine dodatni?
- Czy `unet_resnet` ma dużo masy po stronie ujemnej?
- Czy mediana attention jest stabilnie wyższa niż mediana resblocków?
- Czy runy z niską medianą w konkretnym module mają gorszy `retain_acc` albo FID?

To ostatnie oznacza “porównanie metryk końcowych z rozkładem alignmentu”: zestawiamy końcowe `UA`, `retain_acc`, `FID`, `CLIP` z tym, jak wyglądał rozkład alignmentu przed treningiem.

## Adaptive shrink: proponowana droga

Nie proponuję od razu włączać adaptive shrink jako domyślnego. DASH estymuje gradienty z ograniczonej liczby batchy. Jeśli adaptive shrink będzie agresywny, może dopasować się do losowej wariancji małej próbki, zamiast do stabilnego sygnału. To jest “odtworzenie noise”: moduł wygląda na konfliktowy tylko dlatego, że akurat taki był skład kilku batchy, a algorytm potraktuje to jako prawdziwą cechę.

Bezpieczna kolejność:

1. Zbieraj per-moduł alignment histogram/CDF dla kilku runów.
2. Dodaj diagnostykę shrink distribution i `delta_norm / weight_norm` per moduł.
3. Porównaj rozkłady z `UA`, `retain_acc`, `FID`, `CLIP`.
4. Dopiero potem dodaj eksperymentalny tryb adaptive shrink.

Minimalny wariant adaptive shrink:

```yaml
dash:
  adaptive_shrink:
    enabled: false
    scope: "module"
    statistic: "p25"
    min_value: 0.05
    max_value: 0.8
```

Przykładowa reguła:

```text
module_min_shrink = clamp(p25_alignment, min_value, max_value)
```

Wariant stabilniejszy powinien uwzględniać confidence, np. stabilność znaku/median alignmentu między batchami.

## Obecny stan NSFW

Ważne: obecna ścieżka `sd_nsfw` nie jest jeszcze tak samo spięta z DASH jak class-forgetting.

`pipeline.py` dla `setting: sd` przekazuje `dash_config` do classowych metod unlearningu. Dla `setting: sd_nsfw` obecne `run_unlearn_nsfw(...)` wywołuje:

- `intact_unlearn_nsfw(...)`, albo
- `nsfw_removal(...)`,

ale nie przekazuje tam `dash_config`, a `intact_unlearn_nsfw(...)` nie wykonuje `run_dash_sd_warm_start(...)`.

To znaczy: jeśli chcesz używać DASH w NSFW, trzeba najpierw jawnie podłączyć DASH do ścieżki NSFW.

## Jak obecnie InTAct robi NSFW unlearning

Obecna ścieżka NSFW z InTAct jest w `intact_unlearn_nsfw(...)` w `SD/train-scripts/intact_unlearn.py`.

Flow:

1. Ładuje bazowy Stable Diffusion przez `setup_model(...)`.
2. Tworzy dataloadery przez `setup_forget_nsfw_data(...)`.
3. Definiuje dwa stałe prompty:
   - forget prompt: `"a photo of a nude person"`,
   - retain/pseudo prompt: `"a photo of a person wearing clothes"`.
4. Buduje `UnlearnIntervalProtection` na wskazanych targetach, np. `to_q`, `to_k`, `to_v`.
5. Ogranicza optymalizator do parametrów warstw targetowanych przez InTAct.
6. W każdej iteracji liczy:
   - bazowy NSFW loss,
   - InTAct protection loss,
   - sumę `base_loss + intact_loss`.
7. Po treningu zapisuje model i historię strat.

Ważne: obecny NSFW InTAct nie uruchamia DASH przed treningiem.

### Forget set i retain set w NSFW

`setup_forget_nsfw_data(...)` tworzy dwa zbiory:

- forget set:
  - klasa `NSFW`,
  - ścieżka `nsfw_data_path`,
  - domyślnie logicznie oznacza obrazy NSFW/nude,
  - dataloader: `forget_dl`.

- retain set:
  - klasa `NOT_NSFW`,
  - ścieżka `not_nsfw_data_path`,
  - obrazy non-NSFW / clothed,
  - dataloader: `remain_dl`.

Obie klasy używają `load_dataset(data_path)["train"]` i zwracają tylko `example["image"]` po transformacji. Nie ma tu labeli klas jak w Imagenette.

### Jak retain set jest używany w base loss

Base NSFW loss jest liczony w `compute_nsfw_loss(...)`.

Dla retain obrazów:

```text
remain_images + "a photo of a person wearing clothes"
```

liczony jest zwykły `model.shared_step(...)`. To zachęca model, żeby dalej dobrze modelował bezpieczny/clothed rozkład.

Dla forget obrazów:

```text
forget_images + "a photo of a nude person"
```

model porównuje predykcję z pseudo-celem:

```text
forget_images + "a photo of a person wearing clothes"
```

Konkretnie: dla tego samego obrazu i tego samego noise/timestep, output pod promptem nude ma dopasować się do outputu pod promptem clothed. Pseudo-output jest odcinany przez `.detach()`.

Base loss ma postać:

```text
forget_loss + alpha * remain_loss
```

Czyli retain set działa tu jako jawny składnik zachowujący bezpieczne zachowanie modelu.

### Jak retain set jest używany w InTAct protection

`setup_intact_protection(...)` przekazuje do `UnlearnIntervalProtection.setup_protection(...)`:

```text
forget_dataloader=forget_dl
remain_dataloader=remain_dl
```

Najpierw InTAct zbiera aktywacje z warstw targetowanych przez `targets`, ale podstawowy PCA/forget box buduje na forget dataloaderze. Dla każdej targetowanej warstwy:

1. hook zbiera wejścia do warstwy,
2. aktywacje są centrowane,
3. wykonywane jest SVD,
4. powstaje przestrzeń `U_forget`,
5. w tej przestrzeni liczone są bounds `z_min`, `z_max` z percentyli forget activations.

Retain dataloader jest używany dodatkowo, gdy:

```yaml
intact:
  use_actual_bounds: true
```

Wtedy aktywacje retain są projektowane do tej samej przestrzeni `U_forget`, a `inf_low` / `inf_high` są aktualizowane tak, żeby obejmować również retain data. Innymi słowy retain set rozszerza dopuszczalny interval, w którym InTAct próbuje utrzymać zachowanie targetowanych warstw.

Jeśli `use_actual_bounds: false`, retain dataloader nadal jest przekazywany do setupu, ale bounds są ustawiane przez skalowane forget bounds:

```text
inf_low = z_min - infinity_scale
inf_high = z_max + infinity_scale
```

W aktualnym `pipeline_class.yaml` `use_actual_bounds` jest ustawione na `true`, więc retain set realnie wpływa na interval protection.

### Co robi protection loss

Po setupie InTAct zapamiętuje snapshot parametrów targetowanych warstw. W treningu liczy zmianę:

```text
delta_W = current_weight - snapshot_weight
```

i karze takie zmiany, które:

- przesuwają średnią odpowiedź warstwy,
- interferują z przestrzenią residualną poza forget subspace,
- wypychają odpowiedzi w forget subspace poza dozwolone interval bounds.

Na końcu `compute_protection_loss(...)` mnoży karę przez `lambda_interval`. Jeśli `normalize_protection=true`, loss jest normalizowany przez liczbę warstw.

### Co jest trenowane

InTAct nie ustawia globalnie `requires_grad=False` na nietargetowanych warstwach, żeby nie psuć gradient checkpointingu. Zamiast tego:

1. oznacza parametry targetowanych warstw,
2. `get_trainable_params(...)` zwraca tylko te parametry,
3. tylko one trafiają do optymalizatora.

Targety są dopasowywane substringiem po nazwach modułów lub po nazwie typu modułu. Przykład: `to_q` dopasuje moduły, których nazwa zawiera `to_q`.

## Jak podejść do DASH dla NSFW

NSFW różni się od class-forgetting:

- forget set to obrazy NSFW,
- retain set to obrazy non-NSFW / clothed,
- prompty są semantycznie bardzo silne: np. “nude person” vs “person wearing clothes”,
- ryzyko degradacji ogólnej jakości i ludzi/clothing jest większe niż w class-forgetting Imagenette.

Rekomendowana strategia:

1. Najpierw podłączyć DASH jako warm-start do `intact_unlearn_nsfw(...)`.
   - Użyć `setup_forget_nsfw_data(...)` jako forget/retain loaderów.
   - Użyć opisów/promptów NSFW i clothed jako `descriptions`.
   - Uruchomić `run_dash_sd_warm_start(...)` przed `setup_intact_protection(...)` albo przynajmniej przed właściwym optimizer loop.

2. Zacząć od narrow attention target:
   ```yaml
   dash:
     target: "unet_xattn"
     signal_mode: "retain_only"
     plasticity_granularity: "per_filter"
     include_bias: false
   ```

3. Porównać z:
   ```yaml
   target: "unet_attn"
   ```

4. Dopiero potem próbować:
   ```yaml
   target: "unet"
   ```

5. Nie zaczynać od biasów.

6. Dla NSFW szczególnie monitorować:
   - NSFW detection / UA,
   - CLIP na unsafe prompts,
   - FID/probe na clothed/not-NSFW,
   - sample images,
   - per-moduł attention alignment CDF,
   - retain quality na promptach z ludźmi ubranymi.

## Proponowane rozszerzenia kodu

Najbliższe sensowne kroki:

1. Podłączyć `dash_config` do `run_unlearn_nsfw(...)`.
2. Dodać `dash_config` argument do `intact_unlearn_nsfw(...)`.
3. Uruchomić `run_dash_sd_warm_start(...)` w NSFW przed treningiem.
4. Dodać NSFW-specific naming/logging dla DASH, tak jak w class-forgetting.
5. Dodać opcjonalny target head-aware dla attention.
6. Dodać shrink histogram/CDF oraz `delta_norm / weight_norm` per moduł.

## Praktyczne rekomendacje na teraz

Dla class-forgetting:

- jeśli chcesz sprawdzić attention tylko przez prompt/koncepcję:
  ```yaml
  target: "unet_xattn"
  ```

- jeśli chcesz attention szerzej:
  ```yaml
  target: "unet_attn"
  ```

- jeśli chcesz attention + resblocki obecnym kodem:
  ```yaml
  target: "unet"
  ```

Dla NSFW:

- najpierw podłączyć DASH do ścieżki NSFW,
- zacząć od `unet_xattn`,
- potem `unet_attn`,
- unikać biasów na start,
- nie włączać adaptive shrink przed zebraniem diagnostyki per moduł.

## Aktualizacja: DASH attention head-wise oraz NSFW RL/ROFT

Zaimplementowano rozdzielenie obsługi attention od `plasticity_granularity`. Nowa opcja `dash.attention_head_wise` domyślnie ma wartość `false`, więc dotychczasowe konfiguracje zachowują stare znaczenie: attention używa `global` albo `per_filter` dokładnie tak jak pozostałe moduły. Gdy `attention_head_wise: true`, zwykłe moduły `Linear`/`Conv2d` nadal używają `plasticity_granularity`, a standardowe projekcje attention `Linear.weight` są dzielone na jednostki odpowiadające głowom.

Dla `to_q`, `to_k` i `to_v` głowa oznacza ciągły blok wierszy macierzy wag `W[h_start:h_end, :]`, bo w konwencji PyTorch `Linear.weight` ma kształt `[out_features, in_features]`, a głowy są sklejone po osi wyjściowej. Dla `to_out` głowa oznacza ciągły blok kolumn `W[:, h_start:h_end]`, ponieważ sklejony wymiar głów jest wejściem projekcji wyjściowej. Kod rozpoznaje również wariant CompVis `to_out.0`. Jeśli warstwa nie jest standardowym `nn.Linear.weight`, jest biasem, ma niezgodny kształt, nie ma możliwego do odczytania rodzica attention albo liczby głów nie da się bezpiecznie wywnioskować, DASH wraca do `plasticity_granularity` i zlicza fallback w statystykach `dash_sd_attention_headwise_fallback_*`.

Logowanie globalne DASH zostało zachowane. Dodatkowo zapisywane są per-modułowe `delta_norm` i `relative_delta_norm`, gdzie `delta_W = W_after_dash - W_before_dash`, a `relative_delta_norm = ||delta_W|| / (||W_before_dash|| + eps)`. Dla attention dodano osobne histogramy/CDF/medianę alignmentu dla grup z `dash.attention_logging.groups`, domyślnie `attn2.to_q`, `attn2.to_k`, `attn2.to_v`, `attn2.to_out`. Nie dodano adaptive shrink ani gęstych metryk shrink dla attention.

DASH został też podłączony do ścieżki `sd_nsfw` dla metod `rl` i `roft`. Integracja jest w `random_label.py` przez `certain_label_nsfw(...)` i `retain_only_finetune_nsfw(...)`, a `pipeline.py` wybiera te funkcje tylko dla `unlearn.method: rl` albo `unlearn.method: roft`. `intact_unlearn_nsfw(...)` nie zostało podłączone do DASH. Kolejność w NSFW RL/ROFT jest następująca: załadowanie Stable Diffusion, zbudowanie `forget_dl` z `nsfw_data` i `remain_dl` z `not_nsfw_data`, ewentualny `run_dash_sd_warm_start(...)`, a dopiero potem `model.train()`, wybór parametrów, konstrukcja optymalizatora i zwykła pętla unlearningu. DASH nie wywołuje `optimizer.step()` i modyfikuje tylko wybrane wagi U-Netu.

NSFW DASH używa na start prostego trybu `dash.nsfw.loss_mode: denoise`. Forget gradienty liczone są z obrazów NSFW i promptu `a photo of a nude person`, retain gradienty z obrazów non-NSFW i promptu `a photo of a person wearing clothes`. Tryb `nsfw_base` pozostaje TODO i celowo zgłasza błąd, jeśli zostanie włączony przed implementacją.



## Aktualizacja: diagnostyka DASH i metryki NSFW

Po audycie NSFW RL/ROFT doprecyzowano dwie rzeczy w logowaniu:

1. NudeNet w final eval i train eval zawsze raportuje grupowane liczniki `Common`, `Female`, `Male` oraz `Total`. Jeśli `nudenet.detailed: true`, szczegółowe kategorie nadal są logowane, ale grupy tabelkowe są dodawane obok nich, również per `eval_subset_group`.
2. Na końcu final eval logowany jest blok `Generated image directories and artifacts:`. Ma on wskazywać katalog finalnych obrazów I2P/subset, katalog probe NSFW, ewentualny katalog COCO, ścieżki `final_metrics.json` oraz oczekiwany checkpoint diffusers.

DASH SD już wcześniej miał per-modułowe CDF alignmentu oraz per-modułowe `delta_norm` i `relative_delta_norm`. Wartości `relative_delta_norm` są liczone jako:

```text
relative_delta_norm = ||W_after_dash - W_before_dash|| / max(||W_before_dash||, eps)
```

Jeśli wiele modułów ma bardzo podobne `relative_delta_norm`, nie musi to oznaczać błędu. Przy jednorodnej saturacji shrinku, np. `min_shrink=0.8`, naturalnie pojawia się relatywna zmiana około `0.2` w wielu modułach. Do diagnozowania tego efektu przydatne są dodatkowe statystyki shrinku, które zostały przeniesione do DASH w klasyfikacji: `shrink_mean`, `shrink_std`, `shrink_min`, `shrink_max` i `min_shrink_saturation_fraction` per moduł.

Dodano config `SD/configs/pipeline_nsfw_dash.yaml`. Rekomendowana kolejność eksperymentów:

1. NSFW RL baseline: `unlearn.method: rl`, `dash.warm_start: false`.
2. NSFW RL + DASH: `unlearn.method: rl`, `dash.warm_start: true`, `target: unet_xattn`, `signal_mode: retain_only`, `attention_head_wise: false`, `min_shrink: 0.8`.
3. NSFW ROFT baseline: `unlearn.method: roft`, `dash.warm_start: false`.
4. NSFW ROFT + DASH: `unlearn.method: roft`, `dash.warm_start: true`, ten sam target i shrink.
5. Później opcjonalnie: `attention_head_wise: true`.

