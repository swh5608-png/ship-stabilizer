import time
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# =========================================================
# 1. 프로그램 기본 설정
# =========================================================

st.set_page_config(
    page_title="Ship Wave Simulator",
    page_icon="🌊",
    layout="wide"
)

st.title("R/V Gunnerus 3D 자동 파도 시뮬레이터")
st.caption(
    "가상 해양환경에서 파도와 선박의 운동을 실시간으로 확인하는 프로그램"
)

# =========================================================
# 2. 파도 상태별 기본 조건
# =========================================================

SEA_STATES = {
    "잔잔한 파도": {
        "Hs": 0.5,
        "Tp": 4.0,
        "amplitude": 0.8,
        "direction": 0,
        "components": 3
    },

    "거친 파도": {
        "Hs": 2.0,
        "Tp": 7.0,
        "amplitude": 2.0,
        "direction": 25,
        "components": 5
    },

    "폭풍 파도": {
        "Hs": 4.0,
        "Tp": 10.0,
        "amplitude": 3.5,
        "direction": 45,
        "components": 7
    }
}

# =========================================================
# 3. 파도 상태 선택
# =========================================================

st.sidebar.header("해양환경 설정")

sea_state = st.sidebar.selectbox(
    "파도 상태 선택",
    list(SEA_STATES.keys())
)

preset = SEA_STATES[sea_state]

st.sidebar.markdown("---")
st.sidebar.subheader("파도 조건")

custom = st.sidebar.toggle(
    "파도 조건 직접 변경",
    value=False
)

if custom:

    hs = st.sidebar.slider(
        "유의파고 Hs (m)",
        0.2, 5.0,
        float(preset["Hs"]),
        0.1
    )

    tp = st.sidebar.slider(
        "피크 주기 Tp (초)",
        3.0, 14.0,
        float(preset["Tp"]),
        0.5
    )

    direction = st.sidebar.slider(
        "파도 진행 방향 (도)",
        0, 360,
        int(preset["direction"]),
        5
    )

    wave_amplitude = st.sidebar.slider(
        "시각화 진폭 배율",
        0.2, 2.0, 1.0, 0.1
    )

else:

    hs = preset["Hs"]
    tp = preset["Tp"]
    direction = preset["direction"]
    wave_amplitude = 1.0

# =========================================================
# 4. 자동 재생 설정
# =========================================================

st.sidebar.markdown("---")
st.sidebar.subheader("시뮬레이션 제어")

playing = st.sidebar.toggle(
    "자동 재생",
    value=True
)

speed = st.sidebar.slider(
    "시간 진행 속도",
    0.25, 3.0, 1.0, 0.25
)

if "sim_time" not in st.session_state:
    st.session_state.sim_time = 0.0

if st.sidebar.button("시뮬레이션 초기화"):
    st.session_state.sim_time = 0.0

# =========================================================
# 5. 파도 성분 생성
# =========================================================

def create_wave_components(tp, hs, count, seed):

    rng = np.random.default_rng(seed)

    fp = 1.0 / tp

    freqs = np.linspace(
        max(0.04, fp * 0.45),
        fp * 2.0,
        count
    )

    phases = rng.uniform(
        0,
        2 * np.pi,
        count
    )

    # PM 스펙트럼 형태를 이용한 성분 분포
    spectrum = (
        freqs ** -5
        * np.exp(
            -1.25 * (fp / freqs) ** 4
        )
    )

    df = freqs[1] - freqs[0] if count > 1 else 0.1

    amps = np.sqrt(
        2 * spectrum * df
    )

    # 원하는 시각화 파고 수준에 맞춰 정규화
    target_scale = hs / 4.0

    amps = amps / max(
        np.sqrt(np.sum(amps ** 2) / 2),
        1e-9
    ) * target_scale

    directions = (
        np.deg2rad(
            np.linspace(-18, 18, count)
        )
        + np.deg2rad(0)
    )

    return freqs, amps, phases, directions


# =========================================================
# 6. 가상 바다 표면 계산
# =========================================================

def calculate_ocean(
    t,
    hs,
    tp,
    direction,
    wave_amplitude,
    count
):

    x = np.linspace(-42, 42, 55)
    y = np.linspace(-25, 25, 40)

    X, Y = np.meshgrid(x, y)

    freqs, amps, phases, spread = (
        create_wave_components(
            tp,
            hs,
            count,
            seed=123
        )
    )

    Z = np.zeros_like(X)

    theta0 = np.deg2rad(direction)

    for f, a, phase, spread_angle in zip(
        freqs,
        amps,
        phases,
        spread
    ):

        omega = 2 * np.pi * f

        k = omega ** 2 / 9.81

        theta = theta0 + spread_angle

        wave_phase = (
            k * (
                X * np.cos(theta)
                + Y * np.sin(theta)
            )
            - omega * t
            + phase
        )

        Z += (
            a
            * np.cos(wave_phase)
        )

    Z *= wave_amplitude

    return X, Y, Z


# =========================================================
# 7. 선박 형상 생성
# =========================================================

def make_ship_geometry():

    # 선박 길이 방향
    xs = np.linspace(-18.1, 18.1, 35)

    vertices = []

    for x in xs:

        q = x / 18.1

        half = 4.8 * np.sqrt(
            max(
                0.01,
                1 - abs(q) ** 2.3
            )
        )

        vertices.extend([
            [x, -half, 1.15],
            [x, -half * 0.88, -0.25],
            [x, 0, -1.35],
            [x, half * 0.88, -0.25],
            [x, half, 1.15]
        ])

    vertices = np.array(vertices)

    I, J, K = [], [], []

    for i in range(len(xs) - 1):

        for j in range(4):

            a = i * 5 + j
            b = a + 1
            c = (i + 1) * 5 + j
            d = c + 1

            I.extend([a, b])
            J.extend([b, d])
            K.extend([c, c])

    return vertices, np.array(I), np.array(J), np.array(K)


def make_box(cx, cy, cz, lx, ly, lz):

    V = []

    for x in [cx-lx/2, cx+lx/2]:

        for y in [cy-ly/2, cy+ly/2]:

            for z in [cz-lz/2, cz+lz/2]:

                V.append([x, y, z])

    faces = [
        (0,1,3),(0,3,2),
        (4,6,7),(4,7,5),
        (0,4,5),(0,5,1),
        (2,3,7),(2,7,6),
        (0,2,6),(0,6,4),
        (1,5,7),(1,7,3)
    ]

    return (
        np.array(V),
        np.array([f[0] for f in faces]),
        np.array([f[1] for f in faces]),
        np.array([f[2] for f in faces])
    )


HULL = make_ship_geometry()

STRUCTURES = [
    make_box(2, 0, 2.25, 11, 5.4, 2.2),
    make_box(4.3, 0, 4.05, 5.8, 4.8, 1.7),
    make_box(-1.7, 0, 3.65, 1, 1, 2.5),
    make_box(-1.7, 0, 5.15, 2.2, 0.35, 0.25),
    make_box(-5.5, 0, 2.75, 1.8, 1.5, 2.8)
]

# =========================================================
# 8. 선박 운동 계산
# =========================================================

def calculate_ship_motion(
    t,
    hs,
    tp,
    direction,
    wave_amplitude
):

    freqs, amps, phases, spreads = (
        create_wave_components(
            tp,
            hs,
            5,
            123
        )
    )

    theta0 = np.deg2rad(direction)

    roll = 0.0
    pitch = 0.0
    heave = 0.0

    for f, a, phase, spread in zip(
        freqs,
        amps,
        phases,
        spreads
    ):

        omega = 2 * np.pi * f

        k = omega**2 / 9.81

        theta = theta0 + spread

        angle = -omega*t + phase

        # 선박 중심부 파도 높이
        heave += a * np.cos(angle)

        # 파도 경사에 따른 가상 선박 Roll
        roll += (
            -a * k * np.sin(angle)
            * np.sin(theta)
        )

        # 파도 경사에 따른 가상 선박 Pitch
        pitch += (
            -a * k * np.sin(angle)
            * np.cos(theta)
        )

    roll_deg = np.rad2deg(roll * 0.75)

    pitch_deg = np.rad2deg(pitch * 0.75)

    return (
        roll_deg,
        pitch_deg,
        heave
    )


# =========================================================
# 9. 회전 행렬
# =========================================================

def rotation_matrix(roll_deg, pitch_deg):

    r = np.deg2rad(roll_deg)
    p = np.deg2rad(pitch_deg)

    Rx = np.array([
        [1,0,0],
        [0,np.cos(r),-np.sin(r)],
        [0,np.sin(r),np.cos(r)]
    ])

    Ry = np.array([
        [np.cos(p),0,np.sin(p)],
        [0,1,0],
        [-np.sin(p),0,np.cos(p)]
    ])

    return Ry @ Rx


# =========================================================
# 10. 3D 선박 및 파도 생성
# =========================================================

def create_scene(t, hs, tp, direction, wave_amplitude, count):

    X, Y, Z = calculate_ocean(
        t,
        hs,
        tp,
        direction,
        wave_amplitude,
        count
    )

    roll, pitch, heave = calculate_ship_motion(
        t,
        hs,
        tp,
        direction,
        wave_amplitude
    )

    R = rotation_matrix(
        roll,
        pitch
    )

    fig = go.Figure()

    # 바다
    fig.add_trace(
        go.Surface(
            x=X,
            y=Y,
            z=Z,
            colorscale=[
                [0, "#042f55"],
                [0.5, "#087eae"],
                [1, "#a8e8f0"]
            ],
            opacity=0.95,
            showscale=False,
            name="Ocean"
        )
    )

    # 선박 선체
    V, I, J, K = HULL

    hull = (R @ V.T).T

    hull[:, 2] += heave

    fig.add_trace(
        go.Mesh3d(
            x=hull[:,0],
            y=hull[:,1],
            z=hull[:,2],
            i=I,
            j=J,
            k=K,
            color="#123f70",
            name="Ship Hull",
            flatshading=True
        )
    )

    # 흰색 상부 구조물
    for n, box in enumerate(STRUCTURES):

        V, I, J, K = box

        transformed = (R @ V.T).T

        transformed[:,2] += heave

        fig.add_trace(
            go.Mesh3d(
                x=transformed[:,0],
                y=transformed[:,1],
                z=transformed[:,2],
                i=I,
                j=J,
                k=K,
                color="#f5f5f0",
                flatshading=True,
                name="Ship Structure" if n == 0 else None,
                showlegend=(n == 0)
            )
        )

    fig.update_layout(
        height=680,
        margin=dict(l=0,r=0,t=10,b=0),
        scene=dict(
            xaxis_title="Ship Length (m)",
            yaxis_title="Ship Width (m)",
            zaxis_title="Height (m)",
            aspectmode="manual",
            aspectratio=dict(
                x=1.8,
                y=1.1,
                z=0.6
            ),
            camera=dict(
                eye=dict(
                    x=1.7,
                    y=1.6,
                    z=1.1
                )
            ),
            bgcolor="#eaf5fb"
        ),
        showlegend=True
    )

    return fig, roll, pitch, heave, X, Y, Z


# =========================================================
# 11. 실시간 데이터 그래프
# =========================================================

if "history" not in st.session_state:

    st.session_state.history = []

if "last_update" not in st.session_state:

    st.session_state.last_update = time.perf_counter()


# =========================================================
# 12. 실시간 시뮬레이션 화면
# =========================================================

@st.fragment(run_every="150ms")
def simulation_display():

    now = time.perf_counter()

    elapsed = now - st.session_state.last_update

    st.session_state.last_update = now

    if playing:

        st.session_state.sim_time += (
            min(elapsed, 0.5) * speed
        )

    t = st.session_state.sim_time

    count = preset["components"]

    fig, roll, pitch, heave, X, Y, Z = create_scene(
        t,
        hs,
        tp,
        direction,
        wave_amplitude,
        count
    )

    # 실시간 수치 표시
    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Roll",
        f"{roll:.2f}°"
    )

    c2.metric(
        "Pitch",
        f"{pitch:.2f}°"
    )

    c3.metric(
        "Heave",
        f"{heave:.3f} m"
    )

    c4.metric(
        "진행 시간",
        f"{t:.1f} s"
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        key="live_ocean"
    )

    # 그래프 데이터 누적
    history = st.session_state.history

    history.append({
        "time": t,
        "roll": roll,
        "pitch": pitch,
        "heave": heave
    })

    if len(history) > 200:

        history = history[-200:]

    st.session_state.history = history

    history_df = pd.DataFrame(history)

    if len(history_df) > 1:

        left, right = st.columns(2)

        with left:

            st.markdown("### Roll 변화")

            st.line_chart(
                history_df.set_index("time")[
                    ["roll"]
                ],
                height=220
            )

        with right:

            st.markdown("### Pitch 변화")

            st.line_chart(
                history_df.set_index("time")[
                    ["pitch"]
                ],
                height=220
            )


# =========================================================
# 13. 현재 선택된 파도 조건 표시
# =========================================================

st.subheader("선택된 해양환경")

a, b, c, d = st.columns(4)

a.metric("파도 상태", sea_state)
b.metric("유의파고", f"{hs:.1f} m")
c.metric("피크 주기", f"{tp:.1f} s")
d.metric("파도 방향", f"{direction}°")

st.divider()

st.subheader("실시간 3D 해양환경")

simulation_display()

# =========================================================
# 14. 사용 안내
# =========================================================

with st.expander("시뮬레이션 설명"):

    st.markdown("""
    ### 파도 상태

    - 잔잔한 파도: 작은 진폭과 낮은 파도 에너지
    - 거친 파도: 여러 파도 성분에 의한 불규칙 운동
    - 폭풍 파도: 상대적으로 큰 파도 성분과 운동

    ### 선박 운동

    - Roll: 좌우 기울기
    - Pitch: 앞뒤 기울기
    - Heave: 상하 움직임

    ### 주의사항

    본 프로그램은 교육용 가상 시뮬레이션입니다.

    파도는 다중 사인파를 합성하여 생성하며,
    선박 운동은 파도 경사를 이용한 단순화된 가정 모델입니다.

    실제 R/V Gunnerus의 계측값이나
    검증된 선박 운동 예측 결과가 아닙니다.
    """)
