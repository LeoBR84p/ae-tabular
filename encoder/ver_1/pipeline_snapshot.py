# -*- coding: utf-8 -*-
# snapshot gerado automaticamente — não editar manualmente
# gerado em: 2025-10-20T02:40:50.461560

def build_features(
    df_in: pd.DataFrame,
    *,
    save_rejects: bool = False,
    run_dir: Path | None = None,
    tz: str = "America/Sao_Paulo",
    strict_date_threshold: float = 1.0  # em %, erro máximo aceitável antes de levantar exceção
) -> pd.DataFrame:
    """
    Reaplica a engenharia da Etapa 5 sobre `df_in` e retorna o DataFrame de features.

    Parâmetros:
      - save_rejects: se True e run_dir não for None, salva CSV/JSON de registros com data inválida.
      - strict_date_threshold: percentual máximo (em %) de datas inválidas; acima disso levanta exceção.

    Observação:
      - Na Etapa 5 (treino), usamos save_rejects=True para auditoria.
      - Nas Etapas 8+, o padrão (False) evita I/O desnecessário.
    """
    df_feat = df_in.copy()

    # --------- checagem de colunas obrigatórias ----------
    REQ_COLS = ["data_lcto", "username", "lotacao", "contacontabil", "dc", "valormi"]
    missing = [c for c in REQ_COLS if c not in df_feat.columns]
    if missing:
        raise RuntimeError(f"ERRO: colunas obrigatórias ausentes: {missing}")

    # ========= 1) Datas: aaaa-mm-dd com auditoria =========
    def _clean_str(x):
        if pd.isna(x): return ""
        s = str(x).strip()
        s = re.sub(r"\s+", " ", s)
        return s.replace("\ufeff","").replace("\u200b","")

    raw_dates = df_feat["data_lcto"].astype(object).map(_clean_str)
    df_feat["data_lcto_parsed"] = pd.to_datetime(raw_dates, format="%Y-%m-%d", errors="coerce")

    invalid_mask = df_feat["data_lcto_parsed"].isna()
    n_total = len(df_feat)
    n_inv = int(invalid_mask.sum())
    pct_inv = (100.0 * n_inv / n_total) if n_total else 0.0
    print(f"[§5] Qualidade 'data_lcto' (aaaa-mm-dd): inválidos = {n_inv}/{n_total} ({pct_inv:.2f}%).")

    # salvar rejeições (opcional, para treino/auditoria)
    if save_rejects and n_inv > 0 and run_dir is not None:
        stamp = datetime.now(ZoneInfo(tz)).strftime("%Y%m%d-%H%M%S")
        rej_cols = ["data_lcto","username","lotacao","contacontabil","dc","valormi"]
        df_rej = df_feat.loc[invalid_mask, rej_cols].copy()
        (run_dir / f"rejects_data_lcto_{stamp}.csv").write_text(
            df_rej.to_csv(index=False, encoding="utf-8-sig")
        )
        (run_dir / f"rejects_data_lcto_{stamp}.json").write_text(
            df_rej.to_json(orient="records", force_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"[§5] Rejeições salvas: rejects_data_lcto_{stamp}.csv/.json")

    # política de erro (estrita, como na sua etapa): trava se exceder limiar
    if pct_inv > float(strict_date_threshold):
        raise RuntimeError(
            f"[§5] ERRO: {pct_inv:.2f}% datas inválidas (>{strict_date_threshold}%)."
        )

    df_feat["data_lcto"] = df_feat["data_lcto_parsed"]
    df_feat = df_feat.drop(columns=["data_lcto_parsed"]).dropna(subset=["data_lcto"]).reset_index(drop=True)

    # ========= 2) Hierarquia da conta: g1/g2/g3 =========
    conta_digits = (
        df_feat["contacontabil"].astype(str)
               .str.replace(r"\D+", "", regex=True)
    )
    df_feat["conta_digits"] = conta_digits
    df_feat["g1"] = conta_digits.str.slice(0, 1)
    df_feat["g2"] = conta_digits.str.slice(0, 2)
    df_feat["g3"] = conta_digits.str.slice(0, 3)
    df_feat = df_feat[df_feat["g1"].str.len() == 1].reset_index(drop=True)

    # ========= 3) Períodos e valor =========
    df_feat["mes"] = df_feat["data_lcto"].dt.to_period("M").dt.to_timestamp()       # início do mês
    df_feat["trimestre"] = df_feat["data_lcto"].dt.to_period("Q").dt.start_time     # início do trimestre
    df_feat["ano"] = df_feat["data_lcto"].dt.year.astype("int16")
    df_feat["mes_num"] = df_feat["data_lcto"].dt.month.astype("int8")
    df_feat["tri_num"] = df_feat["data_lcto"].dt.quarter.astype("int8")

    df_feat["valormi_float"] = pd.to_numeric(df_feat["valormi"], errors="coerce").fillna(0.0)

    # ========= 4) Helpers (blocos) =========
    def _zblock(values: pd.Series, group_df: pd.DataFrame, group_keys: list[str]) -> pd.Series:
        """
        z-score de 'values' por grupos (group_keys), sem criar colunas no df principal.
        Retorna uma Series alinhada ao índice do df.
        """
        aux = group_df[group_keys].copy()
        aux = aux.assign(_v=values.values)
        grp = aux.groupby(group_keys)["_v"]
        mean = grp.transform("mean")
        std  = grp.transform("std").replace(0.0, np.nan)
        z = (aux["_v"] - mean) / std
        return z.fillna(0.0)

    def make_block(df: pd.DataFrame, keys_base: list[str], period: str, base_name: str, by_dc: bool) -> pd.DataFrame:
        """
        Gera e retorna um bloco (DataFrame) com:
          freq_, val_, val_med_, val__z, val_med__z
        sem tocar no df original.
        """
        keys = keys_base + (["dc"] if by_dc else []) + [period]

        # Frequência
        s_freq = df.groupby(keys)[keys[0]].transform("size").astype("int32")
        # Soma de valor
        s_val  = df.groupby(keys)["valormi_float"].transform("sum")
        # Média (com proteção)
        s_mean = np.where(s_freq > 0, s_val / s_freq, 0.0)

        # Z-scores por entidade (sem período)
        z_keys = keys_base + (["dc"] if by_dc else [])
        s_val_z  = _zblock(pd.Series(s_val, index=df.index),  df, z_keys)
        s_mean_z = _zblock(pd.Series(s_mean, index=df.index), df, z_keys)

        suffix = ("_dc" if by_dc else "")
        cols = {
            f"freq_{period}_{base_name}{suffix}": s_freq,
            f"val_{period}_{base_name}{suffix}": s_val,
            f"val_med_{period}_{base_name}{suffix}": s_mean,
            f"val_{period}_{base_name}{suffix}_z": s_val_z,
            f"val_med_{period}_{base_name}{suffix}_z": s_mean_z,
        }
        return pd.DataFrame(cols, index=df.index)

    # ========= 5) Métricas =========
    NEW_BLOCKS = []

    # Totais agregados por período (atividade geral) — bloco único
    blk_tot = pd.DataFrame(index=df_feat.index)
    blk_tot["freq_mes_user_total"]    = df_feat.groupby(["username","mes"])["username"].transform("size").astype("int32")
    blk_tot["freq_tri_user_total"]    = df_feat.groupby(["username","trimestre"])["username"].transform("size").astype("int32")
    blk_tot["freq_mes_lotacao_total"] = df_feat.groupby(["lotacao","mes"])["lotacao"].transform("size").astype("int32")
    blk_tot["freq_tri_lotacao_total"] = df_feat.groupby(["lotacao","trimestre"])["lotacao"].transform("size").astype("int32")
    NEW_BLOCKS.append(blk_tot)

    # (A) Usuário × Conta × D/C
    NEW_BLOCKS.append(make_block(df_feat, ["username","conta_digits"], "mes",       "user_conta", by_dc=True))
    NEW_BLOCKS.append(make_block(df_feat, ["username","conta_digits"], "trimestre", "user_conta", by_dc=True))

    # (B) Lotação × Conta × D/C
    NEW_BLOCKS.append(make_block(df_feat, ["lotacao","conta_digits"], "mes",       "lotacao_conta", by_dc=True))
    NEW_BLOCKS.append(make_block(df_feat, ["lotacao","conta_digits"], "trimestre", "lotacao_conta", by_dc=True))

    # (C) Hierarquia por USUÁRIO — sem D/C e com D/C
    for level in ["g1","g2","g3"]:
        NEW_BLOCKS.append(make_block(df_feat, ["username", level], "mes",       f"user_{level}", by_dc=False))
        NEW_BLOCKS.append(make_block(df_feat, ["username", level], "trimestre", f"user_{level}", by_dc=False))
        NEW_BLOCKS.append(make_block(df_feat, ["username", level], "mes",       f"user_{level}", by_dc=True))
        NEW_BLOCKS.append(make_block(df_feat, ["username", level], "trimestre", f"user_{level}", by_dc=True))

    # (D) Hierarquia por LOTAÇÃO — sem D/C e com D/C
    for level in ["g1","g2","g3"]:
        NEW_BLOCKS.append(make_block(df_feat, ["lotacao", level], "mes",       f"lotacao_{level}", by_dc=False))
        NEW_BLOCKS.append(make_block(df_feat, ["lotacao", level], "trimestre", f"lotacao_{level}", by_dc=False))
        NEW_BLOCKS.append(make_block(df_feat, ["lotacao", level], "mes",       f"lotacao_{level}", by_dc=True))
        NEW_BLOCKS.append(make_block(df_feat, ["lotacao", level], "trimestre", f"lotacao_{level}", by_dc=True))

    # ========= 6) Concatenação única =========
    if NEW_BLOCKS:
        df_feat = pd.concat([df_feat] + NEW_BLOCKS, axis=1)

    # Ordenação e tipos
    df_feat = df_feat.sort_values(["data_lcto","username","conta_digits"]).reset_index(drop=True)
    for c in [c for c in df_feat.columns if c.startswith("freq_")]:
        df_feat[c] = df_feat[c].astype("int32")

    return df_feat


def encode_categoricals(df: pd.DataFrame,
                        cat_cols: list[str] = None,
                        maps: dict[str, dict[str, int]] = None,
                        suffix: str = "_int") -> pd.DataFrame:
    """
    Projeta colunas categóricas para índices inteiros usando os mapas congelados.

    Parâmetros:
    - df: DataFrame de entrada
    - cat_cols: lista de colunas categóricas (default = CAT_COLS)
    - maps: dicionário de mapas (default = categorical_maps deste run)
    - suffix: sufixo para colunas codificadas (default: '_int')

    Saída:
    - DataFrame com novas colunas <col><suffix> (inteiros).
    """
    if cat_cols is None:
        cat_cols = CAT_COLS
    if maps is None:
        maps = categorical_maps

    out = df.copy()
    for col in cat_cols:
        if col not in out.columns:
            raise ValueError(f"Coluna '{col}' não encontrada no DF para codificar.")
        cmap = maps.get(col)
        if cmap is None:
            raise ValueError(f"Mapa não encontrado para coluna '{col}'.")
        oov = cmap.get("__oov__", 0)
        series = out[col].fillna("").astype(str)
        out[col + suffix] = series.map(lambda x: cmap.get(x, oov)).astype("int32")
    return out

