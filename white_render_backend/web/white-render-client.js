/** Browser integration client. Keep the H3 key only on the backend.
 * Pass the short-lived platform token issued to this project; do not embed a
 * deployment-wide editor token in a published website or localStorage.
 */
export class SyntheticClient {
  constructor(baseUrl, project, token) {
    this.base = baseUrl.replace(/\/$/, "");
    this.prefix = `${this.base}/api/projects/${encodeURIComponent(project)}/`;
    this.token = token;
  }
  async request(path, { method = "GET", body, key } = {}) {
    const headers = { Authorization: `Bearer ${this.token}` };
    if (body !== undefined) headers["Content-Type"] = "application/json";
    if (key) headers["Idempotency-Key"] = key;
    const response = await fetch(this.prefix + path, {
      method, headers, body: body === undefined ? undefined : JSON.stringify(body),
    });
    const result = await response.json();
    if (!response.ok) throw Object.assign(new Error(result.error?.message || "请求失败"), result);
    return result;
  }
  capabilities() { return this.request("synthetic/capabilities"); }
  spec(description, appearance_description = "") {
    return this.request("synthetic/specs", { method: "POST", body: { description, appearance_description } });
  }
  async upload(file) {
    const data = new FormData(); data.append("file", file);
    const response = await fetch(this.prefix + "uploads", {
      method: "POST", headers: { Authorization: `Bearer ${this.token}` }, body: data,
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error?.message || "上传失败");
    return this.request("synthetic/assets", { method: "POST", body: { upload_id: result.upload_id } });
  }
  preflight(reference) { return this.request("synthetic/preflight", { method: "POST", body: reference }); }
  quote(reference) { return this.request("synthetic/quotes", { method: "POST", body: reference }); }
  submit(body, key) {
    if (!key) throw new Error("请为此用户操作保存固定幂等键；网络重试继续使用同一键");
    return this.request("synthetic/jobs", { method: "POST", body, key });
  }
  job(id) { return this.request(`jobs/${encodeURIComponent(id)}`); }
  asset(id) { return this.request(`synthetic/assets/${encodeURIComponent(id)}`); }
  cancel(id) { return this.request(`jobs/${encodeURIComponent(id)}/cancel`, { method: "POST" }); }
  resume(id) { return this.request(`jobs/${encodeURIComponent(id)}/retry`, { method: "POST" }); }
  export(asset_id, key, kind = "available") {
    return this.request("synthetic/exports", { method: "POST", key, body: { asset_id, kind, purpose: "research" } });
  }
  async mediaUrl(assetId, filename) {
    const link = await this.request(`synthetic/assets/${encodeURIComponent(assetId)}/media-link`, {
      method: "POST", body: { filename, expires_seconds: 300 },
    });
    return { url: this.base + link.url, expiresAt: link.expires_at_epoch };
  }
}
