declare module "*.mjs" {
  const createModule: (options?: Record<string, unknown>) => Promise<unknown>;
  export default createModule;
}
