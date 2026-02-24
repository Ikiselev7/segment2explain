import { Layout } from "./components/Layout";
import { useWebSocket } from "./hooks/useWebSocket";
import "./styles/index.css";

function App() {
  const { send } = useWebSocket();
  return <Layout onSend={send} />;
}

export default App;
