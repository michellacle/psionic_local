#ifndef PSIONIC_MLX_CAPI_H
#define PSIONIC_MLX_CAPI_H

#ifdef __cplusplus
extern "C" {
#endif

char *psionic_mlx_capi_compatibility_scope_json(void);
char *psionic_mlx_capi_compatibility_matrix_json(void);
char *psionic_mlx_capi_eval_json(const char *request_json);
void psionic_mlx_capi_string_free(char *value);

#ifdef __cplusplus
}
#endif

#endif
