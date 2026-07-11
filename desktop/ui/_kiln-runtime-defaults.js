(() => {
  const serverHost = "127.0.0.1";
  const serverPort = 8420;
  globalThis.KILN_RUNTIME_DEFAULTS = Object.freeze({
    serverHost,
    serverPort,
    serverBaseUrl: `http://${serverHost}:${serverPort}`,
    openAiBaseUrl: `http://${serverHost}:${serverPort}/v1`,
  });
})();
