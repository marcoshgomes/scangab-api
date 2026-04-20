import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------
# 🔍 1. LOCALIZAR ÂNCORAS
# ---------------------------------------------------
def find_anchors(img_gray):
    blur = cv2.GaussianBlur(img_gray, (5, 5), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    anchors = []
    for c in cnts:
        area = cv2.contourArea(c)
        if 1000 < area < 100000:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.04 * peri, True)
            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(c)
                if 0.8 < (w / float(h)) < 1.2:
                    M = cv2.moments(c)
                    if M["m00"] != 0:
                        anchors.append([int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])])

    return np.array(anchors) if len(anchors) >= 4 else None

# ---------------------------------------------------
# 🔍 2. DETECTAR BOLHAS (Agora com filtro Y para Cabeçalho/Rodapé)
# ---------------------------------------------------
def detectar_bolhas(thresh_img, debug_img):
    cnts, _ = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bolhas = []

    for c in cnts:
        area = cv2.contourArea(c)
        if 150 < area < 4000:
            (x, y), r = cv2.minEnclosingCircle(c)
            posX, posY = int(x), int(y)
            
            # 🔥 FILTRO DE ZONA TOTAL (X e Y):
            # Ignora as margens laterais (X)
            is_na_largura_correta = (110 < posX < 460) or (540 < posX < 900)
            
            # 🔥 NOVO FILTRO VERTICAL: Ignora cabeçalho (até Y=250) e rodapé (após Y=1320)
            is_na_altura_da_grade = (250 < posY < 1320)

            if is_na_largura_correta and is_na_altura_da_grade and 8 < r < 25:
                bolhas.append((posX, posY, int(r)))
                # Azul para bolinhas detectadas na zona de questões
                cv2.circle(debug_img, (posX, posY), int(r), (255, 0, 0), 2)
            else:
                # Amarelo para tudo o que foi ignorado (textos, logos, números)
                cv2.circle(debug_img, (posX, posY), int(r), (0, 255, 255), 1)

    return bolhas

# ---------------------------------------------------
# 🔍 3. ORGANIZAR QUESTÕES
# ---------------------------------------------------
def organizar_questoes(bolhas, alternativas=5):
    bolhas_esq = [b for b in bolhas if b[0] < 500]
    bolhas_dir = [b for b in bolhas if b[0] > 500]

    def agrupar_coluna(lista_bolhas):
        lista_bolhas = sorted(lista_bolhas, key=lambda b: b[1])
        linhas = []
        tolerancia_y = 20
        
        for b in lista_bolhas:
            if not linhas:
                linhas.append([b])
                continue
            media_y = np.mean([p[1] for p in linhas[-1]])
            if abs(b[1] - media_y) < tolerancia_y:
                linhas[-1].append(b)
            else:
                linhas.append([b])
        
        for i in range(len(linhas)):
            linhas[i] = sorted(linhas[i], key=lambda b: b[0])
        return linhas

    questoes_col1 = agrupar_coluna(bolhas_esq)
    questoes_col2 = agrupar_coluna(bolhas_dir)

    return questoes_col1 + questoes_col2

# ---------------------------------------------------
# 🔥 4. LER RESPOSTAS
# ---------------------------------------------------
def ler_respostas(linhas_de_bolhas, thresh_img, gray_img, debug_img, alternativas=5):
    respostas = []

    for linha in linhas_de_bolhas:
        if len(linha) < alternativas:
            respostas.append(None)
            continue

        linha = linha[:alternativas]
        scores = []

        for (x, y, r) in linha:
            r_crop = int(r * 0.8)
            roi_bin = thresh_img[y - r_crop : y + r_crop, x - r_crop : x + r_crop]
            roi_gray = gray_img[y - r_crop : y + r_crop, x - r_crop : x + r_crop]

            if roi_bin.size == 0:
                scores.append(0)
                continue

            preenchimento = np.sum(roi_bin) / 255
            contraste = 255 - np.mean(roi_gray)
            scores.append(preenchimento * contraste)

        scores = np.array(scores)
        idx = np.argmax(scores)
        
        # Ajuste de sensibilidade: 25% acima da média para garantir a escolha
        if scores[idx] > np.mean(scores) * 1.25 and scores[idx] > 450:
            respostas.append(["A", "B", "C", "D", "E"][idx])
            cv2.circle(debug_img, (linha[idx][0], linha[idx][1]), linha[idx][2] + 3, (0, 255, 0), 3)
        else:
            respostas.append(None)

    return respostas

# ---------------------------------------------------
# 🚀 5. ROTA DA API
# ---------------------------------------------------
@app.post("/processar")
async def processar_prova(file: UploadFile = File(...), questoes: int = Form(...), alternativas: int = Form(...)):
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        pts = find_anchors(gray)
        if pts is None:
            return {"status": "erro", "mensagem": "Âncoras não detectadas"}

        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1)
        corners = np.array([pts[np.argmin(s)], pts[np.argmin(diff)], pts[np.argmax(s)], pts[np.argmax(diff)]], dtype="float32")

        W, H = 1000, 1400
        M = cv2.getPerspectiveTransform(corners, np.array([[0, 0], [W, 0], [W, H], [0, H]], dtype="float32"))
        warped = cv2.warpPerspective(img, M, (W, H))
        warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        
        thresh = cv2.adaptiveThreshold(warped_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 5)

        debug_img = warped.copy()

        bolhas = detectar_bolhas(thresh, debug_img)
        if len(bolhas) < 10:
            return {"status": "erro", "mensagem": f"Poucas bolhas detectadas ({len(bolhas)})"}

        questoes_agrupadas = organizar_questoes(bolhas, alternativas)
        respostas = ler_respostas(questoes_agrupadas, thresh, warped_gray, debug_img, alternativas)

        # Retorna apenas a quantidade de questões solicitada pelo professor
        respostas_finais = respostas[:questoes]

        cv2.imwrite("debug_final.jpg", debug_img)

        return {
            "status": "sucesso",
            "questoes_detectadas": len(respostas_finais),
            "respostas": respostas_finais
        }
    except Exception as e:
        return {"status": "erro", "mensagem": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)