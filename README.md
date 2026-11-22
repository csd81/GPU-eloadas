# GPU Programozás – 1. Előadás (1. rész) Jegyzet



---

## 1. Adminisztráció, követelmények és fejlesztői környezet

* **Oktatói háttér és nyelv:** Az oktató párhuzamos numerikus algoritmusokkal és fizikai szimulációkkal foglalkozik (alapvetően C++ környezetben). Az angol nyelvű diasorokat magyarul adja elő.
* **Számonkérés:**
* **Aláírás feltétele:** Egy félév végi beadandó feladat elkészítése (választható témákkal, pl. egyszerű adattranszformáció vagy képtanuló algoritmus az AI-irányultságú hallgatóknak).
* **Vizsga:** Papíralapú, írásbeli, kifejtős **elméleti vizsga** a vizsgaidőszakban (nincs helyszíni kódolás).


* **Gyakorlati környezet:** Mivel a kari gépekben nincs dedikált GPU (és a hazai *Komondor* szuperszámítógépes hozzáférést csak a nappali tagozat kapta meg), a gyakorlatok **Google Colab** környezetben futnak (**CUDA Core** és **Numba CUDA** Python-csomagokkal, illetve az elején C++ fordításra is használva a virtuális gépet).
* **Helyi telepítés buktatói:** A teljes helyi **CUDA Toolkit** telepítése saját driver-verziót hoz magával, ami driver-ütközéshez és akár képernyő nélküli rendszerhibához vezethet – csak saját felelősségre ajánlott. Windows alatt a *Nsight Visual Studio Edition* (ahol az IntelliSense hibásan pirossal aláhúzza a `<<<...>>>` kernelhívásokat), Linux alatt a *VS Code + `make*` használatos.
* **Kötelező és ajánlott irodalom:**
* **Shane Cook: *CUDA Programming*** (legfontosabb elméleti alapmű a vizsgához).
* ***CUDA by Example*** (gyakorlati, tutoriál jellegű C++ példatár).
* ***GPU Computing Gems*** (esettanulmányok, sikertörténetek).
* **NVIDIA CUDA Programming Guide** (hivatalos online referencia, külön Pythonos bevezető résszel).



---

## 2. Motiváció és a párhuzamosítás korlátai

* **Fetch–Decode–Execute ciklus korlátai:** Egy hagyományos CPU működése az utasítások betöltéséből, dekódolásából, végrehajtásából és a memóriába történő visszaírásából áll, amelyet az órajel ciklusideje korlátoz.
* **Moore-törvény megtorpanása:** Az órajel nem emelhető tovább a hőtermelés és a fizikai hűtési korlátok miatt, a tranzisztorok sűrítése pedig eléri az atomi lépték határait. A megoldás a **párhuzamosítás** (több műveletvégző egység alkalmazása).
* **FPGAs és célhardverek:** Maximális hatékonyság hardveresen huzalozott FPGA-kkal érhető el, de ezek fejlesztése túl drága és lassú. Az NVIDIA ehelyett általános célú GPU-kba integrál speciális célhardvereket (pl. **Tensor-magok** hardveres mátrixszorzáshoz).
* **Teljesítménymérés (FLOPS):** A számítási teljesítményt lebegőpontos művelet/másodperc (**FLOPS**) egységben mérjük, ahol egy művelet jellemzően egy **FMA** (*Fused Multiply-Add*: $a \times b + c$).
* **Elméleti csúcsteljesítmény:** A hardveregységek számának és maximális sebességének szorzata.
* **Gyakorlati (mért) teljesítmény:** Szintetikus tesztekkel (sorozatos összeadások, FMA-k) mért érték. Valós programoknál már a **tizenegynéhány százalékos** hardverkihasználtság is kiváló eredménynek számít.



---

## 3. Szuperszámítógépek és az Exascale korszak

| Rendszer / Kezdeményezés | Helyszín / Háttér | Teljesítmény és műszaki jellemzők |
| --- | --- | --- |
| **Frontier** | USA (Oak Ridge) | Az első Exascale ($>10^{18}$ FLOPS) szuperszámítógép. |
| **El Capitan** | USA | Korábbi csúcstartó (~2,7 exaFLOPS elméleti csúcs). |
| **Kínai Exascale gép** | Sencsen (Kína) | 2,2 exaFLOPS mért teljesítmény; 100%-ban saját tervezésű GPU-k, hálózati összeköttetés és operációs rendszer (megtörve a nyugati hardvermonopóliumot). |
| **JUPITER** | Jülich (Németország) | Európa első Exascale gépe (EuroHPC); AI, okosgyárak és *digital twin* projektek számára épült. |
| **Komondor** | Debrecen (Magyarország) | 5 petaFLOPS; félig töltött HPE Cray kabinet **NVIDIA A100** GPU-kkal. |
| **Levente** | Budapest, KFKI (tervezett) | Előkészített infrastruktúrával rendelkező jövőbeli magyar rendszer. |

* **Exascale szűk keresztmetszet:** Míg ~10–15 petaFLOPS-ig a feladat egyszerű szétosztása is jól skálázódik, Exascale szinten a több ezer GPU közötti **hálózati kommunikációs késleltetés** (a rengeteg switchen és hálózati elemen átfutó adatátvitel akkumulálódása) válik a fő korláttá.

---

## 4. Párhuzamos végrehajtási stratégiák és szoftveres paradigmák

### Végrehajtási modellek (Flynn-taxonómia és csővezeték)

1. **SIMD (*Single Instruction, Multiple Data*):** Egyetlen utasítással egy vektor több elemén hajtunk végre ugyanazt a műveletet ciklus írása nélkül (pl. Intel CPU **AVX** utasításkészlet).
2. **SIMT (*Single Instruction, Multiple Threads*):** A CUDA végrehajtási modellje, amely az SIMD szálalapú kiterjesztése (32 szálas csoportokban, *warp*-okban hajtja végre ugyanazt az utasítást).
3. **MIMD (*Multiple Instruction, Multiple Data*):** Klasszikus konkurens programozás, ahol eltérő szálak vagy folyamatok különböző feladatokat végeznek különböző adatokon (pl. C++ `std::thread`), ami komoly szinkronizációs kihívásokat vet fel.
4. **Pipelining (futószalagos feldolgozás):** Az egymást követő munkafázisok (adatbeolvasás, számítás, kiírás) időbeli átfedése eltérő hardveregységeken – a CUDA-ban a **streamek** alapja.

### Szoftveres paradigmák

* **OpenMP (közös memóriás / *Shared Memory*):** Többmagos CPU-k közös memóriájára épül. Fordítói direktívákkal (`#pragma omp`) párhuzamosítja a ciklusokat; megfelelő szinkronizáció nélkül versenyhelyzet (*memory corruption*) léphet fel.
* **MPI (*Message Passing Interface* – elosztott memóriás):** Olyan rendszerekhez és szuperszámítógépes fürtökhöz használják, ahol a csomópontok nem látnak bele egymás memóriájába. Szabványos pont-pont és **kollektív** üzenetküldési műveletekkel szinkronizálja az adatcserét (pl. GPU-fürtökön a CPU-k MPI-vel koordinálják a csomópontok közötti kommunikációt).
* **OpenCL:** Heterogén rendszerek (CPU + GPU) egységes programozására szolgáló keretrendszer, de az automatizált fordítás teljesítményveszteséggel jár.

---

## 5. CPU vs. GPU architektúra és GPGPU

* **GPGPU (*General-Purpose computing on GPU*):** A GPU-kat eredetileg 3D-s csúcspontok (*vertexek*) transzformálására és képpontok árnyalására (*shading*, *blending*) tervezték. A kutatók felismerték, hogy ezek mátrix- és lineáris algebrai műveletek, így a grafikus futószalag „eltérítésével” (*hijacking*) tudományos számításokra is befoghatók.
* **Platformok:** A kurzus az NVIDIA **CUDA** modelljét használja; az AMD kártyák ekvivalens megoldása a **HIP** (amely szintaktikailag szinte azonos, csak a `cuda` előtagot `hip` váltja fel).

| Szempont | CPU (*Host*) | GPU (*Device*) |
| --- | --- | --- |
| **Tervezési cél** | **Alacsony késleltetés (*Low Latency*)** – egyetlen feladat minél gyorsabb befejezése. | **Magas áteresztőképesség (*High Throughput*)** – minél több művelet egyidejű elvégzése. |
| **Chipfelület** | Kevés (4–16), rendkívül komplex mag, nagy vezérlőlogika és hatalmas gyorsítótár (*cache*). | Több száz/ezer egyszerűsített **ALU** (alapműveletekre optimalizálva), jóval kevesebb *cache*. |
| **Önállóság** | Önállóan vezérli saját működését (utasításmutatók, operációs rendszer). | **Alárendelt célhardver**, önálló működésre képtelen; a CPU vezérli a **PCI Express** buszon keresztül. |

---

## 6. Élő demonstrációk és teljesítménybeli tanulságok

1. **N-body (többtest-probléma) szimuláció (*CUDA Samples*):**
* **CPU:** 4096 részecskénél ~20 FPS (6,2 GFLOPS); 16 000 részecskénél 1,3 FPS-re lassul.
* **GPU:** 16 000 részecskénél ~400 FPS (**2,3 TFLOPS**); 65 000 részecskénél **3+ TFLOPS**.
* **Alulterhelés és *Latency Hiding*:** Ha a részecskeszámot visszavesszük 4096-ra, a GPU teljesítménye ~300 GFLOPS-ra esik vissza, mert nincs elég párhuzamos munka az összes végrehajtó egység leterheléséhez. A GPU akkor hatékony, ha „beterítjük” feladattal: ilyenkor a szálcsoportok közötti kontextusváltás (*context switch*) ideje elrejthető (*latency hiding*).


2. **Mandelbrot- és Julia-halmaz megjelenítő:**
* **CUDA–OpenGL interoperabilitás:** A GPU közvetlenül a saját videómemóriájába számolja ki a pixeleket (a CPU-ra való visszamásolás nélkül), így valós idejű nagyítás érhető el több száz FPS mellett (egyszeres pontosságú `float` esetén a mély nagyításnál elfogy a lebegőpontos felbontás).


3. **Argon plazma ionizációs szimuláció (KFKI együttműködés):**
* Nanoszekundumos időskálán zajló 2D plazmafizikai szimuláció, amely hagyományos PC-n **hónapokig** (akár fél évig) futott; a *Komondor* szuperszámítógépen 8 GPU-val **2,5 órára** csökkent a futási idő.



---

## 7. A CUDA programozási modell alapjai

* **Hardverfüggetlenség (PTX és Compute Capability):** A CUDA a különböző GPU-generációkat egy egységes programozási modell és a kártya képességeit jelző **Compute Capability** verziószám mögé rejti. A fordító (`nvcc`) először egy köztes, assembly-szerű kódra (**PTX** – *Parallel Thread Execution*) fordítja a programot, amelyből futáskor a driver állítja elő a konkrét kártyára optimalizált natív gépi kódot.
* **Host és Device kód szétválasztása:** A program két részből áll: a CPU-n futó **host-oldali vezérlőkódból** (amely a CUDA runtime könyvtárakon keresztül memóriát foglal, adatot másol és elindítja a számításokat), valamint a GPU-n párhuzamosan végrehajtódó **kernel függvényekből**.
* **A modell „2+1” pillére:**
1. **Fizikai hardverarchitektúra:** A számítóegységek (ALU-k) elrendezése a chipen.
2. **Szálkezelési hierarchia:** A logikai szálak szervezése és ütemezése.
3. **Memóriahierarchia:** A regiszterek, a chipen belüli gyorsítótárak (beleértve a programozható *shared memory*-t) és a kártyán lévő globális DRAM/VRAM kezelése.


* **A GPU-programozás 3 szintje:**
1. *Drop-in könyvtárak* (kész, optimalizált algoritmusok, pl. FFT, lineáris algebra).
2. *Direktíva-alapú / C++ STL párhuzamosítás* (`nvc++` execution policy).
3. *Alacsony szintű, saját kernelek írása* C/C++ vagy Python nyelven (**a kurzus fő fókusza**).

A **GPU202627 1. előadás** (mindkét rész) törzsanyagát három egymásra épülő pillér fogja össze: a fizikai hardverhierarchia, a logikai szálmodell és a memóriaarchitektúra.

---

## 1. Architektúra és a háromszintű CUDA leképezés

A CUDA programozási modell teljes egészében a fizikai szilícium felépítését tükrözi vissza. A CPU-val szemben (amely alacsony késleltetésre és komplex elágazáskezelésre optimalizált) a GPU magas áteresztőképességre (*throughput*) és tömeges párhuzamosságra épül.

| Hierarchia szintje | Fizikai hardver (GPU) | Logikai szálmodell | Memóriaarchitektúra |
| --- | --- | --- | --- |
| **1. Alapegység** | **CUDA mag (Core / SP)** – ALU-egység (INT, FP, Tensor, SFU) | **Szál (Thread)** – egyetlen adatelem műveletét végzi | **Regiszter (Register)** – szálanként privát, leggyorsabb, szűkös |
| **2. Csoportos egység** | **Streaming Multiprocessor (SM)** – saját vezérlő, programszámláló, regiszterbank | **Szálblokk (Thread Block)** – max. **1024 szál**, 1D/2D/3D elrendezés | **Shared Memory** – az SM-en belül a blokk minden szála közösen eléri (L1-gyel osztozik a fizikai chipen) |
| **3. Teljes rendszer** | **Teljes GPU** (SM-ek és TPC-k sokasága) | **Grid** – az összes blokk összessége (1D/2D/3D) | **Globális memória (VRAM / DRAM)** – minden blokk és szál eléri, nagy kapacitás, nagy késleltetés |

* **Kiegészítő memóriatípusok:**
* **Constant memory (`__constant__`):** Csak olvasható globális memória, amelyhez az SM-en belül dedikált *constant cache* tartozik (a teljes program futása alatt megőrzi a gyorsítótárazott értékeket).
* **Host memória:** A CPU főmemóriája, amelyet a GPU közvetlenül nem ér el; az adatokat a **PCI Express** buszon keresztül kell másolni.



---

## 2. Végrehajtási modell és ütemezés

* **SIMT és Warp végrehajtás:** Egy blokk szálai **32-es csoportokban (warpokban)** képződnek le az SM számítási magjaira (`warpSize = 32`). Egy warpon belül minden szál ugyanazon órajelütésnél pontosan ugyanazt a gépi utasítást hajtja végre (*Single Instruction, Multiple Threads*).
* **Warp divergencia (`if-else` kezelése):** Ha egy warpon belül feltételes elágazás történik, a hardver egy **aktív maszkot** (*active mask*) számol: először végiglépked az `if` ágon az igaz feltételű szálakkal (a többi várakozik), majd invertálja a maszkot, és végiglépked az `else` ágon. Ez kiszekvencializálja a futást, ezért a kernelekben kerülni kell a szálanként eltérő elágazásokat.
* **Kétszintű ütemezés és Latency Hiding:**
* A **Block Scheduler** osztja ki a szálblokkokat az egyes SM-ekre. A blokkok szigorúan függetlenek egymástól (blokkok között nincs szinkronizáció).
* A **Warp Scheduler** az SM-en belül váltogatja a futásra kész warpokat. Ha egy warp memóriára várakozik, az ütemező azonnal átvált egy másik warpra, miközben a regiszterállapotok a helyükön maradnak — ez a **latency hiding**, amelyhez minél több aktív blokkra és szálra van szükség.


* **Teljesítménymetrikák (Roofline szemlélet):** A programok hatékonyságát a számítási teljesítmény (**FLOPS**) és a memória-sávszélesség (**GB/s**) együttesen határozza meg (*compute-bound* vs. *memory-bound* feladatok).

---

## 3. Globális szálindexelési képletek

A GPU memóriájában a többdimenziós struktúrák lineárisan (kilapítva) helyezkednek el. A beépített `dim3` / `uint3` változók (`gridDim`, `blockIdx`, `blockDim`, `threadIdx`) segítségével minden szál saját maga számítja ki, melyik adatelemen dolgozik:

* **1D globális index (vektorokhoz):**

$$\text{idx} = \text{blockIdx.x} \times \text{blockDim.x} + \text{threadIdx.x}$$


* **2D globális koordináták és lineáris memóriaindex (mátrixokhoz):**

$$x = \text{blockIdx.x} \times \text{blockDim.x} + \text{threadIdx.x}$$


$$y = \text{blockIdx.y} \times \text{blockDim.y} + \text{threadIdx.y}$$


$$\text{linear\_idx} = y \times \text{width} + x$$



---

## 4. CUDA C++ kulcsszavak és Runtime API

A **CUDA Runtime API** magasabb szintű absztrakciót nyújt a GPU driver (*Driver API*, pl. `cuLaunchKernel`) felett, és az `nvcc` fordító dolgozza fel (PTX köztes kódra fordítva a kerneleket).

### Függvény- és változómódosítók

* `__global__`: GPU-n futó **kernel** függvény (CPU-ról hívható, visszatérési értéke mindig `void`, Python/Numba megfelelője: `@cuda.jit`).
* `__device__`: Csak GPU-n futó és csak GPU-ról hívható segédfüggvény.
* `__host__`: CPU-n futó függvény (kombinálható: `__host__ __device__` mindkét architektúrára lefordul).
* `__shared__`: Az SM közös memóriájában létrejövő változó/tömb (használatakor blokkon belüli `__syncthreads()` szinkronizáció szükséges a *data race* elkerülésére).
* `__constant__`: Csak olvasható, gyorsítótárazott konstans memória.

### Tipikus CUDA program életciklusa (Host vezérlés)

1. **Memóriafoglalás a GPU-n:** `cudaMalloc((void**)&d_a, size)` (érdemes a `h_` és `d_` előtagokkal elkülöníteni a *host* és *device* pointereket).
2. **Adatok másolása GPU-ra:** `cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice)`
3. **Kernel indítása (aszinkron):**
```cpp
kernel<<<Dg, Db, Ns, S>>>(d_a, d_b);

```


* `Dg`: *Grid* dimenziója (blokkok száma `x, y, z` irányban).
* `Db`: *Block* dimenziója (szálak száma `x, y, z` irányban, szorzatuk $\le 1024$).
* `Ns` *(opcionális)*: Dinamikusan foglalt *shared memory* mérete bájtokban (`extern __shared__` tömbökhöz).
* `S` *(opcionális)*: CUDA *stream* azonosító párhuzamos végrehajtási sorokhoz.


4. **Szinkronizáció és visszamásolás:** `cudaDeviceSynchronize()` (szükség esetén), majd `cudaMemcpy(h_a, d_a, size, cudaMemcpyDeviceToHost)` (az alapértelmezett streamen a másolás implicit módon megvárja a kernel befejezését).
5. **Memória felszabadítása:** `cudaFree(d_a)`.

---

## 5. Gyakorlati környezet (Google Colab & NVCC)

* **Colab beállítás:** A futtatókörnyezetet **T4 GPU**-ra kell állítani (*Tesla T4*, Compute Capability `7.5`, 40 SM, 48 KB *shared memory*/blokk, 65 536 regiszter/blokk). A munkamenet fájlrendszere ideiglenes, ezért tartós munkához célszerű a Google Drive-ot csatolni (*Mount Drive*).
* **Alapvető parancsok:**
* `!nvidia-smi` – elérhető GPU, driver- és CUDA-verzió ellenőrzése.
* `!nvcc -o kimenet forras.cu` – CUDA C++ kód fordítása (a `-arch=sm_75` kapcsolóval megadható a célarchitektúra).


## 1. GPU hardver-, memória- és szálarchitektúra (Alapok)

* **CPU vs. GPU felépítés:** Míg a CPU kevés számú, komplex, alacsony késleltetésű magot tartalmaz, a GPU több ezer egyszerűbb, nagy áteresztőképességre (*throughput*) optimalizált számítási magból áll, amelyek **Streaming Multiprocessorokba (SM)** szerveződnek.
* **SIMT (*Single Instruction, Multiple Threads*) és Warp:**
* A végrehajtás alapegysége a **Warp**, amely mindig **32 szálból** áll. Egy warpon belül közös az utasításmutató (*instruction pointer*), így a 32 szál fizikailag ugyanazt az utasítást hajtja végre egyszerre különböző adatokon.


* **Latency Hiding (késleltetés elrejtése):** Egy **blokk** mindig egyetlen SM-en fut, de egymástól függetlenül ütemezhető. Ha egy blokk memóriaműveletre vár, az SM hardveres ütemezője azonnal átvált egy másik számításra kész blokkra, így a kontextusváltási idő eltűnik.
* **Szinkronizáció:** Egy blokkon belüli szálak a *shared memory*-n keresztül kommunikálhatnak és blokkszinten szinkronizálhatók, de a teljes **grid** szinkronizálása csak a *host* oldalról lehetséges (újabb kernelindítással vagy explicit szinkronizációval).

| Hardveres szint | Szoftveres (CUDA) megfelelő | Memóriatípus és jellemzők |
| --- | --- | --- |
| **CUDA Core (mag)** | **Thread (szál)** | **Regiszterek:** Chipen belüli, leggyorsabb; csak az adott szál látja. |
| **Streaming Multiprocessor (SM)** | **Block (blokk)** (1D/2D/3D) | **Shared Memory & L1 Cache:** Chipen belüli, programozható; a blokk összes szála közösen eléri. |
| **Teljes GPU** | **Grid (rács)** (1D/2D/3D) | **Global, Constant, Texture, Local (DRAM):** Nagy kapacitású (40–80 GB), de lassabb elérésű; az L2 cache gyorsítja. |

---

## 2. Compute Capability és a CUDA fordítási folyamat

* **Compute Capability (`sm_xy`):** Nem azonos az architektúra nevével (pl. Maxwell: `5.2`, Volta: `7.0`, Ampere: `8.6`, Hopper: `9.0`), hanem a hardveres funkciókészlet és a PTX utasítások verziószáma. Meghatározza, milyen modern utasítások érhetők el (pl. hardveres atomi műveletek), és biztosítja a visszafelé kompatibilitást. Ha nem adjuk meg, az `nvcc` gyakran alacsonyabb alapértelmezett verzióra fordít, ami teljesítményvesztést okozhat.
* **Kétágú fordítási lánc (`.cu` forrásból):**
1. **Host ág:** A standard C++ kódot a *host* fordító (pl. `gcc`/`MSVC`) tárgykóddá (`.o`/`.obj`) fordítja. A kernelindításokat a fordító *host*-oldali *stub* függvényhívásokká alakítja, amelyek a CUDA Runtime-on keresztül indítják a GPU-t.
2. **Device ág:** A `cicc` komponens a GPU-kódot **PTX** (*Parallel Thread Execution*) köztes reprezentációra fordítja. A `ptxas` assembler ebből bináris **cubin** állományt készít, amelyet a rendszer egy **Fat Binary**-ba csomagol, majd a linker beágyazza a végső futtatható programba.



| Réteg | Teljes név | Szerepe és jellemzői |
| --- | --- | --- |
| **PTX** | *Parallel Thread Execution* | Dokumentált, alacsony szintű virtuális utasításkészlet (**portability layer**). Biztosítja, hogy egy program újrafordítás nélkül hordozható legyen újabb GPU-generációkra. |
| **SASS** | *Streaming Assembly* | Nem dokumentált, az adott GPU-architektúrára szabott **natív gépi kód**, amelyet a hardver ténylegesen végrehajt (a PTX-ből fordítja a driver vagy a `ptxas`). |

* **C++ hibakezelés:** Minden CUDA Runtime hívás egy `cudaError_t` (`enum`) értékkel tér vissza, emellett a `cudaGetLastError()` függvénnyel kérdezhető le az aktuális hibaállapot.

---

## 3. A CUDA Python ökoszisztéma

A Python egy magas szintű *wrapper* réteget biztosít a C/C++ alapú CUDA eszköztár fölé: gyors prototípusgyártást és alacsonyabb fejlesztési komplexitást ad, cserébe az interpreter futásideje kisebb többletterhelést jelent, illetve a gyenge típusosság miatt fokozottan figyelni kell a tömbök pontos adattípusára (pl. `float32`, előjeles vs. előjel nélküli egészek).

* **`cuda.core`:** A CUDA Runtime objektumorientált Python burkolója (memóriafoglalás, másolás, eszközszinkronizáció).
* **`cuda.compute`:** Előre optimalizált alapvető algoritmusok („GPU Lego elemek”) natív Python környezethez.
* **`Numba` (`numba.cuda`):** JIT (*Just-In-Time*) fordító. A `@cuda.jit` dekorátorral ellátott Python-függvényeket nem interpretálja, hanem közvetlenül **PTX** köztes kóddá fordítja a GPU számára.
* **`CuPy` (`cupy`):** A NumPy GPU-megfelelője; az adatszerkezetek (`cp.ndarray`) közvetlenül a videókártya memóriájában jönnek létre és élnek.
* **RAPIDS (`cuDF`):** A Pandas táblázatkezelő műveletek GPU-val gyorsított megfelelője.

---

## 4. Numba CUDA kernel implementáció (AXPY példa)

Az **AXPY** művelet a BLAS terminológiából származó vektorművelet:


$$x_i = a \cdot x_i + y_i$$

### Kulcsképletek az indexeléshez és méretezéshez

* **Globális 1D szálindex:**

$$i = \text{cuda.blockIdx.x} \cdot \text{cuda.blockDim.x} + \text{cuda.threadIdx.x}$$


* **Szükséges blokkok száma (Grid méret) felső egészrésszel:**

$$\text{grid\_size} = \left\lceil \frac{N}{\text{block\_size}} \right\rceil = \text{int}(\text{math.ceil}(N / \text{block\_size}))$$


* **Memóriahatár-védelem:** Ha $N$ nem osztható maradék nélkül a blokkmérettel (pl. $N=7$, blokkméret $4 \rightarrow 8$ elindított szál), a felesleges szálak túlindexelnék a tömböt. Ezt mindig ki kell védeni a `if i < x.shape[0]:` feltétellel.

### Teljes, működő Numba CUDA kód

```python
import math
from numba import cuda
import numpy as np


# 1. GPU kernel definiálása JIT fordítással
@cuda.jit
def axpy_kernel(a, x, y):
    # Globális 1D szálindex kiszámítása
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

    # Tömbhatár-ellenőrzés (nem egész blokkméretű tömbök védelmére)
    if i < x.shape[0]:
        x[i] = a * x[i] + y[i]


# 2. Host-oldali vezérlő és tesztfüggvény
def kernel_test(n=7, block_size=4):
    try:
        a = np.float32(2.0)
        # Host oldali tömbök létrehozása explicit típussal
        x = np.ones(n, dtype=np.float32)
        y = np.arange(n, dtype=np.float32)

        # Adatok másolása a CPU (host) memóriából a GPU (device) memóriába
        dx = cuda.to_device(x)
        dy = cuda.to_device(y)

        # Grid méret meghatározása felső egészrésszel
        grid_size = int(math.ceil(n / block_size))

        # Aszinkron kernelindítás Python [grid, block] szintaxissal
        axpy_kernel[grid_size, block_size](a, dx, dy)

        # Explicit szinkronizáció (opcionális, ha a másolás előtt bevárjuk a GPU-t)
        cuda.synchronize()

        # Eredmény visszamásolása a GPU memóriából a host memóriába
        x_result = dx.copy_to_host()
        print("Eredmény:", x_result)

    except Exception as e:
        print(f"CUDA futási hiba történt: {e}")


kernel_test(n=7, block_size=4)

```

### Fontos végrehajtási szabályok Pythonban

* **Explicit adatmozgatás:** Egy `ndarray` objektum vagy a CPU-n (`np.ndarray`), vagy a GPU-n (`DeviceNDArray` / `cp.ndarray`) él. A `cuda.to_device()` másolja fel az adatot, a `.copy_to_host()` pedig hozza vissza.
* **Aszinkron futás:** A `kernel[grid, block](...)` hívás azonnal visszatér a CPU-n, miközben a GPU a háttérben számol. A `cuda.synchronize()` bevárja a GPU befejezését (megjegyzés: a `copy_to_host()` önmagában is szinkronizáló művelet).


A Google Colab környezetben a parancssori profilozók (`nvprof`, `nsys`, `ncu`) futtatásához először önálló fájlba kell menteni a kódot, majd `!` előtaggal kiadni a terminálparancsokat.

---

## 1. Előkészítés Google Colabban

A cellák tartalmának fájlba írása (hogy parancssorból futtatható legyen):

```python
%%writefile mxm.py
# Ide kerül a teljes futtatható Python/Numba kód (a main() hívással együtt)

```

C++ CUDA kód fordítása (ha natív `.cu` fájlt profilozol):

```bash
!nvcc -O3 mxm.cu -o mxm

```

*(Megjegyzés: Nsight Compute forráskód + SASS assembly összerendeléséhez a `-lineinfo` kapcsoló is ajánlott: `!nvcc -O3 -lineinfo mxm.cu -o mxm`.)*

---

## 2. `nvprof` (Régebbi, tisztán szöveges parancssori profiler)

Bár újabb CUDA Toolkit csomagokban már elavult (*deprecated*), a Google Colab Tesla T4 példányain még elérhető gyors szöveges ellenőrzéshez.

| Parancs | Mit csinál? |
| --- | --- |
| `!nvprof python mxm.py` | Alapértelmezett futtatás: összegzi a CUDA API hívásokat (`cudaMalloc`, `cudaMemcpy`) és a GPU-aktivitás százalékos megoszlását. |
| `!nvprof --print-gpu-trace python mxm.py` | Részletes GPU trace: kiírja a kernel regiszterhasználatát, a *shared memory* méretét, az áteresztőképességet (*throughput*), valamint a forrás- és célmemória típusát (`Pinned` vs. `Pageable`). |
| `!nvprof --print-api-trace python mxm.py` | A host-oldali CUDA Runtime/Driver API hívások időrendi listázása. |
| `!nvprof ./mxm` | Lefordított natív C++ CUDA bináris profilozása. |

---

## 3. Nsight Systems (`nsys` — Rendszerszintű idővonal)

Az `nvprof` modern utódja a teljes rendszer (CPU folyamatok, memóriamásolások, ütemezési késleltetések és GPU sávok) vizsgálatára.

### Telepítés Colabban (ha nincs előre telepítve)

```bash
!apt-get update && apt-get install -y cuda-nsight-systems-13-0

```

*(A verziószám — pl. `12-6` vagy `13-0` — a Colab aktuális CUDA környezetétől függ; `!nsys --version` paranccsal ellenőrizhető, hogy már elérhető-e.)*

### Profilozás és riportgenerálás

```bash
# Python/Numba szkript profilozása és mentése .nsys-rep fájlba
!nsys profile -o mxm_report python mxm.py

# C++ bináris profilozása felülírással (--force-overwrite=true) és NVTX/CUDA nyomkövetéssel
!nsys profile --trace=cuda,nvtx --force-overwrite=true -o mxm_cpp_report ./mxm

```

### Eredmények megtekintése

* **Grafikus felületen (ajánlott):** A létrejött `mxm_report.nsys-rep` fájlt töltsd le a Colab bal oldali fájlkezelőjéből (vagy `files.download('mxm_report.nsys-rep')` paranccsal), és nyisd meg a saját gépeden futó asztali **Nsight Systems** alkalmazásban.
* **Közvetlenül a Colab termináljában (szöveges összegzés):**
```bash
!nsys stats mxm_report.nsys-rep

```



---

## 4. Nsight Compute (`ncu` — Kernel-szintű mélyelemzés)

Egy adott kernel belső hardveres teljesítményének (L1/L2 cache kihasználtság, memóriasávszélesség, *warp stalling*, SASS assembly) mérésére szolgál *replay* módszerrel.

| Parancs | Mit csinál? |
| --- | --- |
| `!ncu python mxm.py` | Alapértelmezett kernel-szintű gyorsjelentés (*SpeedOfLight* metrikák: Compute vs. Memory throughput) közvetlenül a terminálba. |
| `!ncu --set full -o mxm_ncu_report python mxm.py` | Teljes részletességű hardveres profil (cache, regiszter, warp stall okok) mentése `.ncu-rep` fájlba, amely asztali Nsight Compute-ban megnyitható. |
| `!ncu --kernel-name regex:mxm_kernel python mxm.py` | Csak a megadott nevű kernel mérése (kiszűri a keretrendszer egyéb segédkerneljeit). |
| `!ncu --launch-skip 1 --launch-count 1 python mxm.py` | Átugorja az első (*warm-up* / JIT fordításos) futást, és csak a második kernelindítást méri. |

---

## Gyors döntési segédlet a méréshez

1. **Hol akad el a program az idővonalon? (Sok üresjárat, lassú `cudaMalloc` vagy másolás)** $\rightarrow$ `nsys profile -o riport python script.py`
2. **Miért lassú maga a kernel számítás közben? (Memóriasávszélesség, regiszterek, warp elakadások)** $\rightarrow$ `ncu --set full -o riport python script.py`
3. **Csak egy 3 másodperces szöveges ellenőrzés kell Colabban letöltés nélkül?** $\rightarrow$ `nvprof --print-gpu-trace python script.py` vagy `nsys stats riport.nsys-rep`


# A **GPU 20251017 class3 1** előadás teljes vizsgafelkészítő jegyzete, a fő elméleti pillérek és a gyakorlati hisztogram-implementációk mentén rendszerezve:

## 1. Shared Memory (Osztott memória) három nézőpontból

| Nézőpont | Jellemzők és szerep |
| --- | --- |
| **Hardveres szint** | **On-chip** (szilíciumon belüli), fizikailag közvetlenül az SM (Streaming Multiprocessor) számítási magjai mellett, az **L1 adat-cache** szintjén helyezkedik el. Ampere architektúrán tipikusan **192 KB** oszlik meg dinamikusan (driverből állítható arányban) az L1 cache és a shared memory között. Elérése mindössze néhány órajelciklus. |
| **Programozási modell** | **Blokk-szintű** (*thread block*): egy adott blokk összes szála szabadon írhatja és olvashatja, de **más blokkok nem látják**. Blokkok közötti adatcseréhez a globális memórián keresztül kell másolni. |
| **Teljesítmény (Lokalitás)** | Programozható cache-ként kihasználja az **időbeli lokalitást** (*temporal locality* — gyakran újrahasznált adatok közel tartása) és a **térbeli lokalitást** (*spatial locality* — pl. rács-/szomszédsági számításoknál a *cache miss*-ek csökkentése). |

### Allokációs módok és szinkronizáció

* **Statikus allokáció:** Fordítási időben ismert méret esetén a kernelen belül deklarálható: `__shared__ float buffer[64];`. A kernelhívás szintaxisa nem változik.
* **Dinamikus allokáció:** Futásidőben megadott méret esetén a kernelben `extern __shared__ int bins_shared[];` szerepel, a méretet pedig **bájtban** kell átadni a kernelindítás 3. konfigurációs paramétereként: `kernel<<<grid, block, num_bins * sizeof(int)>>>(...)`. Egy kernelen belül egyetlen egybefüggő dinamikus puffer foglalható (több tömb esetén manuális *offset*-elés szükséges).
* **Blokkszintű szinkronizáció (`__syncthreads()`):** Megakadályozza a blokkon belüli *data race*-eket. Minden szálnak el kell érnie a korlátot (*barrier*), mielőtt továbbhaladnak.
* **Kritikus szabályok:** A teljes *grid* globálisan **nem szinkronizálható** (ez biztosítja a blokkok független ütemezhetőségét). Továbbá `__syncthreads()` **tilos** feltételes elágazás (`if`) egyik ágába helyezni, mert a másik ágra futó szálak kikerülik, ami örökös elakadáshoz (*deadlock*) vezet.



---

## 2. Atomikus műveletek (`atomic*`)

* **Működés:** Hardveresen támogatott, oszthatatlan *Read-Modify-Write* műveletek, amelyek versenyhelyzet (*race condition*) nélkül módosítanak közös memóriaterületet.
* **Támogatás:** Kepler architektúrától globális memóriában, **Maxwell architektúrától pedig shared memory-ban is** hardveresen gyorsítottak (32 és 64 bites szavakra).
* **Kulcsfüggvény (`atomicAdd`):** `old = atomicAdd(addr, val);` — egyetlen atomi lépésben hozzáadja `val`-t a `*addr` címen lévő értékhez, és visszaadja a módosítás előtti értéket. (További műveletek: `atomicSub`, `atomicExch`, `atomicMin`, `atomicMax`, `atomicCAS`, bitműveletek).

---

## 3. GPU profilozó eszközök és teljesítményelvek

| Eszköz | Fókusz és használat |
| --- | --- |
| **`nvprof`** | Parancssori profilozó (Nsight Systems elődje), Google Colabban is azonnal futtatható (`nvprof ./program`). Az `--print-gpu-trace` kapcsolóval időrendi táblázatot ad a memóriaműveletekről és kernelekről. |
| **Nsight Systems** | **Rendszerszintű idővonal** (*system-wide timeline*): a CPU-folyamatok, API-hívások (`cudaMalloc`, `cudaMemcpy`) és GPU-kernelek közötti rések, indítási késleltetések (*launch overhead*) vizsgálatára. Python API-val automatizálható. |
| **Nsight Compute** | **Kernel-szintű mélyelemzés**: SM-kihasználtság, *Roofline-modell*, *throughput* metrikák és PTX/SASS *disassembly* utasításonkénti szűk keresztmetszetek kereséséhez. |

* **Mikor éri meg GPU-ra vinni egy feladatot?:** Ha a hasznos kernel-számítás ideje elég hosszú ahhoz, hogy elfedje a host oldali *launch overhead*-et (~250 µs), a lassú `cudaMalloc` foglalást és a `cudaMemcpy` adatátvitelt (*latency hiding*, cél a 80–90%+ GPU kernel-arány).

---

## 4. Esettanulmány: Hisztogramszámítás (CPU vs. Naiv GPU vs. Shared Memory GPU)

A feladat *embarrassingly parallel* jellegű, mert a bemeneti tömb elemei függetlenül sorolhatók be az egyenlő szélességű vödrökbe (*bin*-ekbe).

### Közös vödörindex-számító függvény (`__host__ __device__`)

A `__host__ __device__` minősítőkkel mindkét architektúrára lefordul, a `__forceinline__` pedig megszünteti a függvényhívási költséget:

```cpp
__host__ __device__ __forceinline__ int calc_bin(float item, float min, float max, float bin_width) {
    return (int)((item - min) / bin_width);
}

```

### A két GPU-implementáció összehasonlítása

1. **Naiv globális memóriás verzió (`histogram_kernel_naive`):**
* Minden szál egy bemeneti elemet dolgoz fel (`idx = blockIdx.x * blockDim.x + threadIdx.x`), és közvetlenül a lassú **globális memóriában** növeli a számlálót: `atomicAdd(bins + calc_bin(...), 1);`.
* **Hátrány:** Az összes blokk összes szála ugyanazokat a globális memóriacímeket támadja (*high atomic contention*), ami sorba állítja a memóriaműveleteket.


2. **Shared Memory alapú verzió (`histogram_kernel_shared`):**
* **1. fázis (Inicializálás):** Az első `num_bins` darab szál lenullázza a blokk saját `extern __shared__ int bins_shared[]` tömbjét, majd `__syncthreads()` szinkronizálja a blokkot.
* **2. fázis (Lokális gyűjtés):** Minden szál a gyors **shared memory-ban** végez atomikus növelést (`atomicAdd(bins_shared + bin_idx, 1)`), így a versengés csak a blokkon belüli szálakra korlátozódik.
* **3. fázis (Visszaírás globálisba):** Az első `num_bins` darab szál egy-egy `atomicAdd(bins + threadIdx.x, bins_shared[threadIdx.x])` művelettel összesíti a blokk részeredményét a globális `bins` tömbbe.
* **Eredmény:** Nagy bemeneti tömb (pl. 100 000 elem) és több *bin* (pl. 20) esetén a globális atomikus ütközések drasztikus csökkenése miatt jelentős gyorsulást ér el a naiv kernelhez képest.


# Vizsgára felkészítő összefoglaló a **GPU 20251017 class3 2**   előadás teljes anyagából.

## 1. Kollektív műveletek és a redukció alapjai

* **Kollektív műveletek:** Olyan párhuzamos algoritmikus minták, amelyekben minden szál (vagy folyamat) részt vesz egy közös számítási feladatban (pl. MPI-ban az adatszétosztás és visszagyűjtés).
* **Redukció (Reduction):** Egy $N$ elemű tömb elemeinek egyetlen skalár értékké történő összevonása valamilyen asszociatív bináris művelettel (pl. összegzés/sorösszeg, globális minimum/maximum keresése).
* **Szekvenciális futásidő:** $\mathcal{O}(N)$ egyetlen `for` ciklussal.
* **Párhuzamos (fa-alapú) futásidő:** $\mathcal{O}(\log_2 N)$ lépésszám, mivel minden iterációs szinten megfeleződik az aktív elemek száma.


* **CUDA-specifikus korlátok és kihívások:**
* Nincs általános, teljes tömbre működő beépített könyvtári primitív; a szálak csak **blokkon belül** szinkronizálhatók (`__syncthreads()`), a blokkok között nincs közvetlen hardveres bevárakozás egyetlen kernelhíváson belül.
* A beépített warp-szintű redukció (`__reduce_add_sync` stb.) kizárólag **32 elemre** (egy warp méretére) használható.



---

## 2. Redukciós fák és memória-hozzáférési minták

| Megközelítés | Párosítási logika | Hardveres hatás CUDA-ban |
| --- | --- | --- |
| **Interleaved Addressing** (szomszédos párosítás) | A `0.` a `1.`-vel, a `2.` a `3.`-mal stb. művelődik össze, majd lépésenként duplázódik az ugrásköz (`1, 2, 4...`). | **Kedvezőtlen:** Szakadozottá (fragmentálttá) teszi a memóriahozzáférést, rontja a *coalesced memory access*-t, *shared memory*-ban pedig **bank conflict**-ot okoz. |
| **Sequential Addressing** (felezős megközelítés) | A blokkot kettéfelezzük: a felső (bal) fél szálai a `stride = blockDim.x / 2` távolságra lévő alsó (jobb) fél elemeivel művelődnek össze, majd `stride >>= 1`. | **Előnyös:** Egymás melletti szálak egymás melletti memóriacímeket olvasnak (*coalesced access*), és minimalizálja a *shared memory* bankkonfliktusokat. |

---

## 3. Kétlépcsős redukciós architektúra (Multi-block megoldás)

Mivel a blokkok között nincs globális szinkronizáció, a teljes tömb redukcióját két egymást követő kernelhívásra bontjuk (kihasználva, hogy ugyanabban a CUDA streamben két egymás után indított kernel garantáltan szekvenciálisan fut le):

1. **1. lépcső (Parciális blokkredukció):** `<<<grid, block>>>` konfigurációval elindulva minden blokk kiszámítja a saját adatszeletének részeredményét, amelyet a `threadIdx.x == 0` szál kiír a globális `d_array_out[blockIdx.x]` tömbbe.
2. **2. lépcső (Végső blokkredukció):** `<<<1, grid>>>` konfigurációval (egyetlen blokkban `grid.x` szállal) ugyanaz a kernel végigredukálja a `d_array_out` tömbben tárolt részeredményeket. A végső összeg a `d_array_out[0]` elembe kerül.

*(Megjegyzés: Alternatívaként a 2. kernel helyett `atomicAdd` is használható a részeredmények összegzésére, míg az egyetlen GPU-szállal végzett ciklus teljesen kerülendő, mert egy lassú CPU-maggá degradálja a GPU-t.)*

---

## 4. Implementációk: Globális vs. Shared Memory

### A) Naiv (globális memóriás) kernel

Minden iterációs lépésben közvetlenül a globális memóriában (`array_in`) olvassa és írja a részeredményeket.

```cpp
__global__ void reduceKernelGlobal(float* array_in, float* array_out) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            array_in[idx] += array_in[idx + stride];
        }
        __syncthreads(); // Kötelező: megelőzi a versenyhelyzetet a következő kör előtt
    }

    if (threadIdx.x == 0) {
        array_out[blockIdx.x] = array_in[idx];
    }
}

```

### B) Shared Memory-s kernel

A globális memóriából blokkonként egyszer betöltjük az adatokat a Streaming Multiprocessorhoz (SM) közeli, gyors *shared memory*-ba, ott végezzük el a $\log_2(\text{blockDim.x})$ lépéses redukciót, majd csak a végeredményt írjuk vissza globális memóriába.

```cpp
__global__ void reduceKernelShared(const float* array_in, float* array_out) {
    extern __shared__ float array_shared[];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // 1. Betöltés globális memóriából shared memory-ba + szinkronizáció
    array_shared[threadIdx.x] = array_in[idx];
    __syncthreads();

    // 2. Redukció gyors on-chip shared memory-ban
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            array_shared[threadIdx.x] += array_shared[threadIdx.x + stride];
        }
        __syncthreads();
    }

    // 3. Blokkszintű részeredmény kiírása globális memóriába
    if (threadIdx.x == 0) {
        array_out[blockIdx.x] = array_shared[0];
    }
}

```

* **Host-oldali indítás (`extern __shared__` esetén):** A kernelkonfiguráció 3. paraméterében bájtokban kell megadni a dinamikus *shared memory* méretét:
* 1. hívás: `reduceKernelShared<<<grid, block, block.x * sizeof(float)>>>(d_in, d_out);`


* 2. hívás: `reduceKernelShared<<<1, grid, grid.x * sizeof(float)>>>(d_out, d_out);`




* **Gyakori buktatók:**
* A `stride >>= 1` biteltolás régebbi fordítókon gyorsabb volt az egész osztásnál (`/= 2`), de a modern fordítók már automatikusan optimalizálják.
* Minden `__syncthreads()` elhagyása adathibához (*race condition*) vezet, mert a szálak (lane-ek) még frissítetlen memóriacellákat olvashatnak ki.
* A demonstrációs kód szép páros / 2-hatvány méreteket feltételez; általános tömbméreteknél határellenőrzésre (`idx < length` és `threadIdx.x + stride < blockDim.x`) van szükség.



---

## 5. Nsight Compute profilozási eredmények (Release mód)

* **Kis problémaméret ($256 \times 64 = 16\,384$ elem):** A *shared memory*-s verzió két kernelhívása összesen **~6,3 $\mu\text{s}$** alatt futott le, és a naiv globális verzió is közel azonos időt produkált (kis terhelésnél a memóriahierarchia előnye még kevésbé dominál).
* **Nagyobb problémaméret ($1024 \times 256 = 262\,144$ elem):**
* Globális memóriás (naiv) kernel: **~45 $\mu\text{s}$**
* Shared memory-s kernel: **~34 $\mu\text{s}$** (~24%-os gyorsulás a globális memóriaforgalom csökkentésével)



---

## 6. További redukciós optimalizációs lépcsők

1. **First Add During Load (első összeadás betöltéskor):** A grid méretét eleve megfelezzük, és a szálak már a globális memóriából a *shared memory*-ba történő másoláskor összeadnak két, adott offsettel eltolt elemet (`sdata[tid] = g_idata[i] + g_idata[i + blockDim.x]`).
2. **Warp-szintű redukció az utolsó 32 elemre:** Amikor `stride <= 32`, már csak egyetlen warp aktív. Mivel a warp szálai SIMT módon szinkronban futnak, nincs szükség `__syncthreads()` hívásokra:
* Régebbi módszer: az utolsó warp utasításainak manuális vagy C++ *template* alapú szétbontása (*unroll last warp*).
* Modern CUDA megoldás: a ciklust csak `stride > 32`-ig futtatjuk, az utolsó 32 elemet pedig egyetlen hardveres `__reduce_add_sync(0xffffffff, val)` művelettel összegezzük.


# A **GPU202627 class4 1** előadás 

---

## 1. A Shared Memory helye a GPU architektúrában és a programozási modellben

* **Hardveres elhelyezkedés:** A *shared memory* fizikailag az egyes **Streaming Multiprocessorokon (SM)** belül helyezkedik el, és közös fizikai tárterületen osztozik az **L1 Data Cache**-sel (pl. NVIDIA A100 SM esetén összesen 192 KB). Míg az L1 cache-t a hardver automatikusan kezeli, a *shared memory* **programozható** (egyéni gyorsítótárként működik), és L1 cache sebességű elérést biztosít.
* **Programozási modell leképezése:** A *shared memory* egy **szálblokkon (*thread block*) belül** létezik – a blokk összes szála tudja írni és olvasni, más blokkok szálai viszont közvetlenül nem érik el.
* *Kivétel (Hopper architektúra / Compute Capability 9.0+):* A **Thread Block Cluster** fogalmával a klusteren belüli szomszédos blokkok megoszthatják egymással a *shared memory*-t anélkül, hogy ki kellene menniük a globális memóriába.


* **Lokálitási elvek (*Locality*):**
* **Időbeli lokalitás (*Temporal locality*):** Gyakran írt/olvasott adatszegmensek SM-hez közel tartása.
* **Térbeli lokalitás (*Spatial locality*):** Egymás melletti adatelemek (pl. mátrixcsempék / *tile*-ok) egyszeri betöltése globális memóriából a *shared memory*-ba, hogy elkerüljük az ismétlődő lassú *off-chip DRAM* hozzáféréseket.


* **Explicit szinkronizáció és Data Race:** Mivel egy *warp* (32 szál) párhuzamosan hajtja végre az utasításokat, szinkronizáció nélkül versenyhelyzet (*data race*) alakul ki (egyik szál felülírja, amit a másik olvasna). Ezt a blokkon belüli explicit szinkronizációval (`__syncthreads()` / `cuda.syncthreads()`) kell kezelni.

---

## 2. Shared Memory foglalás: CUDA C++ vs. Python (Numba CUDA)

| Tulajdonság | CUDA C++ | Python (Numba CUDA) |
| --- | --- | --- |
| **Statikus foglalás** | `__shared__ float buf[SIZE];` a kernelben (fordítási idejű konstans méret) | `buf = cuda.shared.array(shape=SIZE, dtype=np.int32)` a kernelen belül |
| **Dinamikus foglalás** | `extern __shared__ float buf[];` a kernelben + `<<<grid, block, bytes>>>` indításkor | `buf = cuda.shared.array(shape=0, dtype=np.int32)` a kernelben |
| **Kernel indítási szintaxis** | `kernel<<<grid, block, shared_bytes, stream>>>(...)` | `kernel[grid, block, stream, shared_bytes](...)` *(stream és byte méret felcserélve!)* |

* **Fontos Numba sajátosságok:**
* A `cuda.shared.array` kizárólag a `@cuda.jit` kernelen belül hívható meg, a host oldalon nem.
* A Numba a kernelt indításkor PTX köztes kódra fordítja, ezért a statikus foglalásnál nem célszerű kívülről változókat beadni; dinamikus foglalásnál a `shape=0` beállítással, híváskor pedig pl. `arr.nbytes` értékkel adható meg a kívánt bájtméret.



---

## 3. Atomi műveletek (*Atomic Operations*) és hardveres korlátok

* **Működési elv:** Az atomi műveletek hardveresen támogatott, megszakíthatatlan **Read-Modify-Write** instrukciók. Egyetlen szál kizárólagosan lefoglalja az adott memóriacímet, kiolvassa, módosítja és visszaírja az értéket, megakadályozva a *race condition*-t.
* **Elérhető műveletek:**
* **CUDA C++:** `atomicAdd`, `atomicSub`, `atomicExch`, `atomicMin`, `atomicMax`, `atomicInc`, `atomicDec`, `atomicCAS` (aritmetikai és bitenkénti logikai műveletek).
* **Numba CUDA:** `cuda.atomic.add(array, index, value)`, `sub`, `exch`, `min`, `max` (szintén csak kernelen belül).


* **Architekturális fejlődés:**
* **Kepler:** Hardveres támogatás a *globális memórián* (*off-chip DRAM*) belüli atomi műveletekre.
* **Maxwell-től:** Hardveres támogatás kiterjesztve a *shared memory*-n belüli atomi műveletekre is.


* **Teljesítménybeli buktató (*Warp Stalling*):** Ha egy warp több szála ugyanazt a memóriacímet próbálja atomi művelettel módosítani, a szálak sorba állnak, ami *warp stalling*-ot okoz és **szekvenciálissá degradálja** a párhuzamos végrehajtást. Nagyobb adatok összesítésére a hierarchikus redukció hatékonyabb.

---

## 4. Hisztogram-számítás: Elmélet és Numba CUDA implementáció

### Feladatmodell és adatpárhuzamosság

* **Probléma:** Egy $N$ elemű, nem negatív egészeket (`0 .. range-1`) tartalmazó bemeneti tömb elemeinek gyakorisági eloszlását számoljuk ki egy `bins` számlálótömbben (`bins[array[i]] += 1`). Folytonos értékek esetén előzetes vödrözés (*binning*) szükséges.
* **Párhuzamosítási stratégia:** Klasszikus **adatpárhuzamos** feladat: minden bemeneti tömbelemhez egy dedikált GPU-szálat rendelünk ($1:1$ leképezés). Mivel egy blokkban legfeljebb **1024 szál** lehet, a nagyobb tömböket több blokkra osztjuk: $\text{grid} = \lceil N / \text{block} \rceil$.

### Szekvenciális CPU referencia és Globális memóriás GPU kernel

```python
import numpy as np
from numba import cuda

# 1. Szekvenciális CPU referencia
def histogram_host(array, bins):
    for elem in array:
        bins[elem] += 1

# 2. GPU kernel (globális memóriás atomi művelettel)
@cuda.jit
def histogram_kernel(array, bins):
    # Globális szálindex explicit kiszámítása
    idx = cuda.blockDim.x * cuda.blockIdx.x + cuda.threadIdx.x
    
    # Tömbhatár-ellenőrzés (out-of-bounds védelem)
    if idx < array.size:
        value = array[idx]
        # Atomi növelés a globális memóriában lévő hisztogram tömbben
        cuda.atomic.add(bins, value, 1)

# Host oldali előkészítés és futtatás
N = 2000
val_range = 5
array = np.random.randint(0, val_range, size=N, dtype=np.int32)
bins_ref = np.zeros(val_range, dtype=np.int32)
bins = np.zeros(val_range, dtype=np.int32)

histogram_host(array, bins_ref)

# Adatmásolás Host -> Device (cudaMalloc + cudaMemcpy egyben)
d_array = cuda.to_device(array)
d_bins = cuda.to_device(bins)

# Grid és Block méretezés (max 1024 szál/blokk!)
block = 256
grid = int(np.ceil(N / block))

# Kernel indítás és eredmény visszamásolása Device -> Host
histogram_kernel[grid, block](d_array, d_bins)
h_bins = d_bins.copy_to_host()

```

### Vizsgatippek a kódhoz:

1. **Memóriakezelés:** A `cuda.to_device()` a C-s `cudaMalloc` és `cudaMemcpy(..., HostToDevice)` műveleteket egyesíti, míg a `.copy_to_host()` visszahozza az eredményt a RAM-ba.
2. **Túlindexelés elleni védelem:** A `np.ceil(N / block)` miatt az utolsó blokkban felesleges szálak is elindulnak, ezért az `if idx < array.size:` feltétel kötelező.
3. **Következő optimalizációs lépés:** A globális memóriás `cuda.atomic.add` sok szál esetén szűk keresztmetszetet okoz; a megoldás a **privatizáció**, ahol minden blokk a saját gyors *shared memory*-jában vezet egy lokális rész-hisztogramot, és csak a blokk végén (szinkronizáció után) adja hozzá az eredményt atomi művelettel a globális `d_bins` tömbhöz.


# GPU202627 – 4. gyakorlat (2. rész) Vizsgajegyzet


---

## 1. Teljesítménymérés és Shared Memory alapú hisztogram

### Környezet és `nvprof` mérés

* A 2026 októberi Google Colab futtatókörnyezet-frissítés miatt az `nvprof` alapértelmezetten nem működik; a mérésekhez vissza kell állítani a **2026.07-es runtime verziót**.
* **Globális memóriás hisztogram szűk keresztmetszete:** Nagyobb adatméretek és kevés bin (pl. `range = 5`) esetén rengeteg szál próbálja ugyanazokat a globális memóriacímeket frissíteni `atomicAdd` művelettel. Ez szerializálja a memória-hozzáféréseket és rontja a futásidőt.

### Shared Memory optimalizáció elve

A **shared memory** a blokkon belüli szálak által közösen elérhető, programozható on-chip gyorsítótár. A globális versengés csökkentésére **privatizációt (lokális rész-hisztogramokat)** alkalmazunk:

1. **Deklaráció és inicializálás:** A shared memory nem nullázódik automatikusan! A blokk első $K$ szála (`cuda.threadIdx.x < len(bins)`) nullára állítja a blokk lokális hisztogramját, majd kötelező egy `cuda.syncthreads()` barrier, hogy elkerüljük a versenyhelyzetet (*race condition*).
2. **Lokális akkumuláció:** Minden szál a gyors shared memory pufferben hajtja végre az `atomicAdd` műveletet.
3. **Szinkronizáció és globális egyesítés (*merge*):** Egy második `cuda.syncthreads()` után a blokk első $K$ szála hozzáadja a lokális bin értékeket a globális tömbhöz (`cuda.atomic.add(bins, tid, bins_local[tid])`).

> **Kritikus buktató (Indexelés):** A shared memory tömböt **mindig lokális szálindexszel** (`cuda.threadIdx.x`) kell indexelni, soha nem globális indexszel (`cuda.grid(1)`), különben az 1. és későbbi blokkok túlcímeznek a pufferen, vagy inicializálatlanul hagyják azt!

### Dinamikus Shared Memory Numba CUDA-ban

Ha a bin-ek száma futásidőben változik, a tömb méretét `0`-ra állítjuk a kernelben, és híváskor adjuk meg **bájtokban**:

```python
import numba
from numba import cuda

@cuda.jit
def histogram_shared_kernel(data, bins):
    idx = cuda.grid(1)
    tid = cuda.threadIdx.x

    # Dinamikus shared memory deklaráció (méret = 0, híváskor adjuk meg bájtokban)
    bins_local = cuda.shared.array(0, dtype=numba.int32)

    # 1. Inicializálás lokális indexszel + barrier
    if tid < len(bins):
        bins_local[tid] = 0
    cuda.syncthreads()

    # 2. Lokális hisztogram építése shared memoryban
    if idx < data.size:
        bin_idx = data[idx]
        cuda.atomic.add(bins_local, bin_idx, 1)
    cuda.syncthreads()

    # 3. Lokális eredmények egyesítése a globális memóriába
    if tid < len(bins):
        cuda.atomic.add(bins, tid, bins_local[tid])

# Kernel indítása (3. paraméter: stream=0, 4. paraméter: shared mem méret BÁJTBAN)
# histogram_shared_kernel[grid, block, 0, bins.size * 4](d_data, d_bins)

```

---

## 2. Párhuzamos redukció elmélete

A **redukció** egy kollektív művelet, amely egy $N$ elemű tömbből egyetlen skalár értéket állít elő asszociatív bináris operátorral (pl. összegzés, minimum, maximum).

| Megközelítés | Működés | Teljesítmény / Probléma |
| --- | --- | --- |
| **1. Naiv globális `atomicAdd**` | Minden szál egyetlen globális változóhoz adja hozzá az elemét. | **Nagyon lassú:** a szálak sorba állnak, a végrehajtás szekvenciálissá válik. |
| **2. Egyetlen szál (`1 block, 1 thread`)** | Egyetlen GPU mag `for` ciklussal összegzi a tömböt. | **Használhatatlan:** ~0,1%-os GPU kihasználtság; a GPU nem alacsony késleltetésű soros futásra készült. |
| **3. Szomszédos párosítás (*Interleaved*)** | Páros indexű szálak összeadják a szomszédos elemeket (`stride = 1, 2, 4...`). | **Warp divergencia:** az `if/else` elágazás miatt warpon belül szekvencializálódik a végrehajtás. |
| **4. Szekvenciális címzés (*Sequential Addressing*)** | A blokk alsó fele összeadja a felső felét (`stride = blockDim // 2` lefelé felezve). | **Optimális:** aktív szálak egybefüggő blokkban futnak, minimalizálja a warp divergenciát. |

### Miért kötelező a kétlépcsős (hierarchikus) redukció?

* A redukciós fa minden szintje között szinkronizálni kell, hogy a következő lépés már a kész részeredményekkel dolgozzon.
* A CUDA **nem támogat globális, blokkok közötti barriert** (`cuda.syncthreads()` csak egy blokkon belül működik), mivel a blokkok ütemezési sorrendje nem garantált.
* **Megoldás:** Kernelhívások közötti implicit szinkronizáció használata:
1. **1. lépés (Blokkszintű redukció):** A kernel minden blokk elemeit egyetlen részösszegbe redukálja, és kiírja a `d_block_sum[blockIdx.x]` tömbbe.
2. **2. lépés (Globális redukció):** Ugyanezt a kernelt **1 blokkal** (`grid = 1`) újra meghívjuk a `d_block_sum` tömbre, amely előállítja a végső összeget.



---

## 3. Redukciós kernelek implementációja Numba CUDA-ban

### A) Globális memóriás kétlépcsős redukció

Minden fa-szinten közvetlenül a globális memóriából olvas és oda ír vissza. *(Megjegyzés: ez az alapverzió kettőhatvány elemszámra működik tisztán peremfeltétel-kezelés nélkül).*

```python
@cuda.jit
def reduce_kernel(array_in, array_out):
    idx = cuda.grid(1)
    tid = cuda.threadIdx.x
    # KÖTELEZŐ az egész osztás (//), különben float lesz és végtelen ciklusba fut!
    stride = cuda.blockDim.x // 2

    while stride > 0:
        if tid < stride:
            array_in[idx] += array_in[idx + stride]
        stride //= 2
        cuda.syncthreads()

    # A blokk részösszege a blokk első elemében gyűlik össze
    if tid == 0:
        array_out[cuda.blockIdx.x] = array_in[idx]

```

* **In-place tulajdonság:** Mivel a kernel a bemeneti tömb első elemében gyűjti össze a részösszeget, a második híváskor a `d_block_sum[0]` elem is felülíródik a végeredménnyel, így minimális munkaterület-allokációval (*in-place*) is használható.

### B) Optimalizált Shared Memory redukció

A lassú, ismétlődő globális memória-hozzáférések (*off-chip memory round-trips*) kiváltására a blokk betölti az adatokat a shared memoryba, ott végzi el a teljes redukciós fát, és csak a végeredményt írja ki a globális memóriába.

```python
@cuda.jit
def reduce_shared_kernel(array_in, array_out):
    idx = cuda.grid(1)
    tid = cuda.threadIdx.x

    # Dinamikus shared memory puffer deklarálása
    shared = cuda.shared.array(0, dtype=numba.int32)

    # 1. Adatok betöltése globális memóriából lokális indexszel
    shared[tid] = array_in[idx]
    cuda.syncthreads()

    # 2. Fa-redukció kizárólag a gyors shared memoryban (lokális indexekkel!)
    stride = cuda.blockDim.x // 2
    while stride > 0:
        if tid < stride:
            shared[tid] += shared[tid + stride]
        stride //= 2
        cuda.syncthreads()

    # 3. Blokk részösszegének kiírása globális memóriába
    if tid == 0:
        array_out[cuda.blockIdx.x] = shared[0]

# Kétlépcsős indítás (int32 = 4 bájt/elem):
# 1. lépés: reduce_shared_kernel[grid, block, 0, block * 4](d_array, d_block_sum)
# 2. lépés: reduce_shared_kernel[1, grid, 0, grid * 4](d_block_sum, d_result)

```

---

## 4. További optimalizációk és beépített absztrakciók

* **Haladó hardveres optimalizációk (Mark Harris – NVIDIA):** A saját shared memory kernel tovább gyorsítható olyan technikákkal, mint a bank konfliktusok elkerülése, az első összeadás elvégzése már a globális memóriából történő betöltéskor (feleannyi blokk indítása), valamint az utolsó 32 szál (1 warp) ciklusának kibontása (*warp unrolling* – szinkronizáció nélkül, mert a warp szálai *lockstep*-ben futnak).
* **C++ `Cooperative Groups` (`cg::reduce`):** Modern C++ CUDA-ban rugalmas szálcsoportok (blokk, csempe vagy warp-részhalmaz) hozhatók létre. A `cg::reduce(block, val, cg::plus<int>())` függvény hardveres warp shuffle (`__shfl_sync`) utasításokkal hajtja végre a redukciót.
* **Python Numba `@cuda.reduce`:** Magas szintű dekorátor, amely egy bináris műveletből automatikusan teljes GPU redukciót generál, kezelve a memóriaallokációt és az adatmozgatást is:

```python
@cuda.reduce
def easy_reduce(a, b):
    return a + b

total = easy_reduce(d_array)

```

* **Mikor melyiket használjuk?:** Önálló tömbösszegzésre a `@cuda.reduce` kényelmes és gyors. Azonban minden absztrakciónak van teljesítménybeli ára (*abstraction cost*), és ha a redukciót egy komplexebb számítási folyamatba (*kernel fusion*) kell integrálni, a saját shared memory alapú implementáció elengedhetetlen.


# A **GPU202627 class5 1** 

---

## 1. Párhuzamos teljesítménymetrikák és gyorsulási törvények (0:00–16:30)

### Alapfogalmak és gyorsulás ($S_P$)

* **Végrehajtási idők:** A méréseket azonos konfiguráción és tesztadaton kell elvégezni:
* $T_{\text{seq}}$ ($T_1$): szekvenciális végrehajtási idő.
* $T_{\text{par}}$ ($T_P$): párhuzamos végrehajtási idő $P$ darab végrehajtási egységen (magok / GPU-k száma).


* **Gyorsulás (*Speedup*, $S_P$):**

$$S_P = \frac{T_{\text{seq}}}{T_{\text{par}}}$$


* **Ideális eset:** Ha nincs párhuzamosítási veszteség (*overhead*), akkor $T_{\text{par, ideális}} = \frac{T_{\text{seq}}}{P}$, amiből az elméleti maximális gyorsulás $S_{P, \text{ideális}} = P$. A valóságban ez sosem érhető el a szálindítás, szinkronizáció, kommunikáció és eredmény-összefésülés (*merge*) költségei miatt.

### Amdahl törvénye (fix problémaméret)

* Legyen $\alpha \in [0, 1]$ a program **eredendően szekvenciális** (nem párhuzamosítható) hányada, $(1 - \alpha)$ pedig a párhuzamosítható része.
* A párhuzamos futási idő ekkor:

$$T_{\text{par}} = \alpha \cdot T_{\text{seq}} + \frac{(1 - \alpha) \cdot T_{\text{seq}}}{P}$$


* **Elméleti gyorsulási korlát ($P \to \infty$):**

$$S_{\max} = \lim_{P \to \infty} \frac{T_{\text{seq}}}{\alpha \cdot T_{\text{seq}} + \frac{(1 - \alpha) \cdot T_{\text{seq}}}{P}} = \frac{1}{\alpha}$$


* **Következmény:** A program szekvenciális hányada alapvetően behatárolja a skálázhatóságot; sőt, a magok/egységek számának túlzott növelésével a kommunikációs és szinkronizációs overhead miatt a futásidő akár romolhat is.

### Gustafson törvénye (skálázódó problémaméret)

* A valóságban $\alpha$ nem konstans, hanem a problémaméret ($N$) függvénye: $\alpha(N)$.
* Ha a feladat méretét növeljük ($N \to \infty$), a párhuzamosan végezhető munka dominálni kezd, így a szekvenciális arány nullához tart: $\lim_{N \to \infty} \alpha(N) = 0$.
* Behelyettesítve visszakapjuk, hogy nagy problémaméreteknél a gyorsulás megközelítheti az ideális $P$ értéket (pl. $N$-test szimulációnál sok részecske esetén).

### GPU-specifikus teljesítménymutatók

* **Miért nem használjuk a klasszikus $S_P$-t egyetlen GPU-n?** Egy GPU több ezer magot tartalmaz, amelyeket a *scheduler* dinamikusan oszt ki az SM-ekre (*Streaming Multiprocessor*), így a pontos aktív $P$ érték nem egyértelmű. (Több-GPU-s rendszerek jellemzésére viszont alkalmas, logaritmikus skálán ábrázolva.)
* **GFLOPS / TFLOPS és Peak Utilization (%):**
* A kártyák névleges csúcsteljesítményét (pl. TFLOPS) ideális, egyszeres pontosságú (`FP32`) **FMA** (*Fused Multiply-Add*) utasítások folyamatos futtatásával mérik.
* Valódi algoritmusoknál a **Peak Performance százalékos kihasználtságát** (*Compute / Memory Peak Utilization*) figyeljük, amely közvetlenül kiolvasható az **Nsight Compute** profilozóból (vagy becsülhető órajelből, magszámból, memóriabusz-szélességből).



---

## 2. Memória-sávszélesség korlát és a Roofline modell (17:00–31:30)

### Memóriakorlát és PCIe adatátvitel

* A legtöbb modern GPU-s alkalmazás **memóriakorlátos (*memory-bound*)**: a számítási egységek gyorsabban dolgoznának, mint ahogy a memóriából az adat beérkezik.
* **Chunk méret hatása:** Kis adatblokkok másolásakor a busz sávszélessége nincs kihasználva; a blokkméret növelésével az átviteli sebesség nő, majd a maximális sávszélességnél ellaposodik (*flat-out*).
* **Host–Device forgalom minimalizálása:** A PCI Express buszon keresztüli másolás rendkívül lassú. A normál (lapozható / *pageable*) host memóriát a CPU előbb átmásolja egy nem lapozható pufferbe, és csak onnan jut el a GPU-ra. Ezzel szemben a **Pinned Memory** (*page-locked*) közvetlenül a hardveres **DMA engine** által másolható, és elengedhetetlen a streamek aszinkron átlapolásához.

### A Roofline teljesítménymodell

A Roofline modell az algoritmus és a hardver viszonyát ábrázolja egy log-log grafikonon:

| Tengely / Elem | Jelentés és mértékegység | Képlet / Magyarázat |
| --- | --- | --- |
| **X-tengely** | **Aritmetikai intenzitás** (*Operational Intensity*) [$\text{FLOP}/\text{Byte}$] | $\text{OI} = \frac{\text{Lebegőpontos műveletek száma (FLOP)}}{\text{Memóriából mozgatott adatmennyiség (Byte)}}$ |
| **Y-tengely** | **Elért számítási teljesítmény** [$\text{GFLOPS}$] | A kernel által ténylegesen produkált műveleti sebesség. |
| **Meredek egyenes** | **Memória-sávszélesség korlát** (*Bandwidth roof*) | Az ez alatti tartomány a **memory-bound** zóna. |
| **Vízszintes vonal** | **Számítási csúcsteljesítmény** (*Compute roof*) | Az ez alatti tartomány a **compute-bound** zóna (külön vonal van `FP32`-re és `FP64`-re). |
| **Töréspont (*Ridge point*)** | **Ideális egyensúlyi pont** | Itt a programot egyforma mértékben korlátozza a memória és a számítás. |

* **Értelmezés és optimalizálás:** A kernelek egy-egy pontként (*marker*) jelennek meg (pl. az **Nsight Compute** automatikusan kirajzolja). A pont fizikailag soha nem lépheti át a tetővonalat.
* *Rossz példa:* Egy egyszerű *copy kernel* nagyon alacsony aritmetikai intenzitású (csak másol, alig számol), így mélyen a *memory-bound* zóna alján helyezkedik el.
* *Jó példa:* Egy optimalizált **cuFFT** vagy összevont **cuFFTDx** kernel magas aritmetikai intenzitással közvetlenül a felső tetővonal alatt fut.



---

## 3. Teljesítményt rontó tényezők és optimalizálási ajánlások (31:30–53:40)

### Fő szűk keresztmetszetek (*Bottlenecks*)

1. **Eredendően szekvenciális algoritmusok és belső függőségek:**
* *Példa:* **Thomas-algoritmus** (tridiagonális mátrixok Gauss-eliminációja). CPU-n $\mathcal{O}(N)$ időben rendkívül gyors, de minden lépés az előző lépés eredményére épül (*internal dependency*), így nem párhuzamosítható. GPU-n egy másik, nagyobb eredendő párhuzamosságot tartalmazó algoritmust kell helyette választani (még ha annak rosszabb is az aszimptotikus lépésszáma).
* **Rekurzió:** GPU-ra vitel előtt iteratívvá kell alakítani / eliminálni kell.
* **Data & Control dependency:** Regiszterben még el nem készült értékekre vagy elágazásokra való várakozás.


2. **Branch Divergence (elágazási divergencia):**
* Egy **warp** 32 szála közös utasításszámlálóval (*Instruction Counter*) egyszerre ugyanazt az utasítást hajtja végre.
* Ha egy `if-else` feltétel mentén a szálak szétválnak, a GPU először az igaz ágat hajtja végre (a többi szál tétlen), majd utána a hamis ágat $\rightarrow$ **implicit szekvencializáció**.
* **Megoldás:** Egymásba ágyazott elágazások kerülése, illetve **branchless programming** (a feltételek $0/1$ értékű szorzóváltozókká alakítása matematikai képletben).
* **Kritikus szabály `__syncthreads()` esetén:** Elágazáson belül a blokkszinkronizációt **minden ágon meg kell hívni**, különben a kimaradó szálak nem érik el a barriert, ami hibás működéshez vagy holtpont-szerű fagyáshoz vezet.


3. **Memóriaelérési mintázatok és bank konfliktusok:**
* **Coalesced memory access:** Egy warp memóriahívása **128 bájtos** tranzakciókban zajlik (pl. $32 \times 4\text{ byte}$ `float`/`int`). Az egymást követő szálaknak egymás után következő memóriahelyeket kell olvasniuk.
* **Global memory bank alignment:** A globális memóriabankok 128 bájtosak; a bankhatárokon átnyúló olvasás lassít.
* **Shared memory bank konfliktus:** A *shared memory* az L1 cache mellett helyezkedik el és **32 memóriabankra** oszlik. Ha egy warp két szála ugyanazt a bankot próbálja egyszerre írni/olvasni, a művelet szekvenciálissá válik (újabb architektúrákon ezt a hardver és a fordító részben enyhíti).


4. **Latencia-problémák:** Ha minden kihasználtsági mutató alacsony, túl sok apró kernelindítás és kis memóriaolvasás késleltetése adódik össze.

### Optimalizálási ökölszabályok

* **Memóriamásolás minimalizálása:** Host–device adatmozgatás visszaszorítása, adat helyben tartása a GPU-n.
* **Kernel Fusing (Kernel Merging):** Több egymást követő kernel összevonása egyetlen kernelbe. Az adatot egyszer olvassuk be a globális memóriából, a köztes eredményeket végig regiszterekben, *shared memory*-ban és L1/L2 cache-ben tartjuk, és csak a végén írjuk ki.
* **Shared memory és Warp-level műveletek:** Lokális adatok gyorsítása *shared memory*-val; warp-szintű kollektív intrinszikus utasítások használata (pl. egyutasításos warp-redukció).
* **Szinkronizáció kerülése:** Globális grid-szinkronizáció (új kernelindítás) és blokkon belüli `__syncthreads()` hívások minimalizálása.
* **Ciklus-kihajtogatás (*Loop unrolling*):** Egymástól független ciklusiterációk kibontása az utasításszintű párhuzamosság (*ILP*) növelésére.
* **Streamek használata:** Konkurens kernel-futtatás és adatátvitel–számítás átlapolása.

---

## 4. Python CUDA memóriakezelés a gyakorlatban (53:40–71:23)

Az egyszerű `cuda.to_device(arr)` hívás minden alkalommal új memóriát foglal és másol is egyszerre. Komolyabb algoritmusoknál a program elején **egyszer kell allokálni**, és futás közben **soha nem szabad dinamikusan újrafoglalni**.

### 1. Numba (Ajánlott, C++ Runtime API-val egyenértékű kontroll)

* **Device foglalás (allokáció másolás nélkül):**
* `d_arr = cuda.device_array(shape, dtype=...)` ($\leftrightarrow$ `cudaMalloc`)
* `d_arr = cuda.device_array_like(h_arr)` – méret és típus átvétele meglévő NumPy tömbből.
* Kényszeríthető a memóriabeli elrendezés: `C_CONTIGUOUS` (sorfolytonos) vagy `F_CONTIGUOUS` (oszlopfolytonos).


* **Másolás meglévő pufferbe (allokáció nélkül):**
* **Host $\to$ Device (H2D):** `cuda.to_device(h_arr, to=d_arr)` ($\leftrightarrow$ `cudaMemcpy` meglévő pointerre). Ha elhagyjuk a `to=` paramétert, új területet is foglal.
* **Device $\to$ Host (D2H):** `d_arr.copy_to_host(h_arr)` – meglévő host pufferbe másol (paraméter nélkül új NumPy tömböt allokál).


* **Pinned (Page-Locked) Memory:**
* `h_pinned = cuda.pinned_array(shape, dtype=...)` vagy `cuda.pinned_array_like(h_arr)` ($\leftrightarrow$ `cudaMallocHost`).
* `cuda.pinned(h_arr)` – meglévő tömb rögzítése (a háttérben másolással jár).


* **Managed Memory (Unified Memory):**
* Automatikus, laponkénti mozgatás a CPU és GPU között. Prototipizálásra kényelmes, de éles teljesítményhez le kell cserélni explicit host/device foglalásra.



### 2. `cuda.core` (Alacsony szintű objektumorientált CUDA Runtime frontend)

* Akkor használjuk, ha a rendszer legmélyebb szintjéig kell lemenni (egyébként a magasabb szintű Numba ajánlott).
* Lépései: `Device` kiválasztása $\to$ `PinnedMemoryResource.allocate(bytes, stream)` (nyers bájtméret és stream megadásával) $\to$ `np.from_dlpack()` segítségével NumPy *array view* készítése $\to$ device `Buffer` foglalása és `.copy_from()` / `.copy_to()` használata.

### 3. CuPy

* Saját *memory poolt* tart fenn mind a host, mind a device oldalon.
* Elsődleges célja, hogy a NumPy jelentős részhalmazát *drop-in replacement* módon, hardveragnosztikus kóddal gyorsítsa GPU-n, nem pedig a kézi memóriakezelés.
