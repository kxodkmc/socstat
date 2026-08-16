//! stdio entry point for the socstat MCP server.
//!
//! Spawned as a subprocess by an MCP client (e.g. Claude, Cursor). Protocol
//! messages are exchanged over stdin/stdout.

use rmcp::ServiceExt;

use socstat_mcp::{SharedState, SocstatMcpServer};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let server = SocstatMcpServer::new(SharedState::arc());
    let running = server.serve((tokio::io::stdin(), tokio::io::stdout())).await?;
    running.waiting().await?;
    Ok(())
}