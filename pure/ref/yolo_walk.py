"""Canonical traversal of yolov8n's layers, in the exact order the pure C++ forward
consumes them. Shared by the unfused exporter and the .pt write-back so both agree
on ordering. Returns a list of module objects; each is either an Ultralytics Conv
(has .conv and .bn -> conv+BN+SiLU) or a plain nn.Conv2d (detect tail, bias, no act)."""

def walk(model):
    L = model.model
    order = []
    def emit(m): order.append(m)
    def c2f(b):
        emit(b.cv1)
        for bott in b.m: emit(bott.cv1); emit(bott.cv2)
        emit(b.cv2)
    def sppf(b): emit(b.cv1); emit(b.cv2)

    emit(L[0]); emit(L[1]); c2f(L[2]); emit(L[3]); c2f(L[4])
    emit(L[5]); c2f(L[6]); emit(L[7]); c2f(L[8]); sppf(L[9])
    c2f(L[12]); c2f(L[15]); emit(L[16]); c2f(L[18]); emit(L[19]); c2f(L[21])
    det = L[22]
    for i in range(3):
        for m in det.cv2[i]: emit(m)
        for m in det.cv3[i]: emit(m)
    return order
