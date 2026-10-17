//! Module for managing breakpoints in a project.
//!
//! Breakpoints are separate from a session because they're not associated with any particular debug session. They can also be set up without a session running.
use anyhow::{Context as _, Result, anyhow};
pub use breakpoints_in_file::{BreakpointId, BreakpointSessionState, BreakpointWithPosition};
use breakpoints_in_file::{BreakpointsInFile, StatefulBreakpoint};
use collections::{BTreeMap, HashMap, HashSet, VecDeque};
use dap::{StackFrameId, client::SessionId};
use futures::{
    StreamExt as _,
    channel::{mpsc, oneshot},
};
use gpui::{App, AsyncApp, Context, Entity, EntityId, EventEmitter, Subscription, Task};
use language::{
    Buffer, BufferEvent, BufferSnapshot, proto::serialize_anchor as serialize_text_anchor,
};
use rpc::{
    AnyProtoClient, TypedEnvelope,
    proto::{self},
};
use std::{
    cell::{Cell, RefCell},
    hash::Hash,
    ops::Range,
    path::Path,
    sync::Arc,
};
use text::{Bias, PointUtf16, Unclipped};
use util::{ResultExt as _, maybe};

use crate::{
    ProjectPath,
    buffer_store::{BufferStore, BufferStoreEvent},
    worktree_store::WorktreeStore,
};

use super::session::ThreadId;

mod breakpoints_in_file {
    use collections::{HashMap, HashSet};

    use super::*;

    #[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
    pub struct BreakpointId(pub(super) u64);

    #[derive(Clone, Debug, PartialEq, Eq)]
    pub struct BreakpointWithPosition {
        pub position: text::Anchor,
        pub bp: Breakpoint,
        pub id: BreakpointId,
    }

    impl BreakpointWithPosition {
        pub fn new(position: text::Anchor, bp: Breakpoint) -> Self {
            Self {
                position,
                bp,
                id: BreakpointId(rand::random()),
            }
        }

        pub(super) fn same_value(&self, other: &Self) -> bool {
            self.position == other.position && self.bp == other.bp
        }
    }

    /// A breakpoint with per-session data about it's state (as seen by the Debug Adapter).
    #[derive(Clone, Debug)]
    pub struct StatefulBreakpoint {
        pub bp: BreakpointWithPosition,
        pub session_state: HashMap<SessionId, BreakpointSessionState>,
    }

    impl StatefulBreakpoint {
        pub(super) fn new(bp: BreakpointWithPosition) -> Self {
            Self {
                bp,
                session_state: Default::default(),
            }
        }
        pub(super) fn position(&self) -> &text::Anchor {
            &self.bp.position
        }
    }

    #[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
    pub struct BreakpointSessionState {
        /// Session-specific identifier for the breakpoint, as assigned by Debug Adapter.
        pub id: u64,
        pub verified: bool,
    }
    #[derive(Clone)]
    pub(super) struct BreakpointsInFile {
        pub(super) buffer: Entity<Buffer>,
        // TODO: This is.. less than ideal, as it's O(n) and does not return entries in order. We'll have to change TreeMap to support passing in the context for comparisons
        pub(super) breakpoints: Vec<StatefulBreakpoint>,
        _subscription: Arc<Subscription>,
    }

    impl BreakpointsInFile {
        pub(super) fn new(buffer: Entity<Buffer>, cx: &mut Context<BreakpointStore>) -> Self {
            let subscription = Arc::from(cx.subscribe(
                &buffer,
                |breakpoint_store, buffer, event, cx| match event {
                    BufferEvent::Saved => {
                        if let Some(abs_path) = BreakpointStore::abs_path_from_buffer(&buffer, cx) {
                            cx.emit(BreakpointStoreEvent::BreakpointsUpdated(
                                abs_path,
                                BreakpointUpdatedReason::FileSaved,
                            ));
                        }
                    }
                    BufferEvent::FileHandleChanged => {
                        let entity_id = buffer.entity_id();

                        if buffer
                            .read(cx)
                            .file()
                            .is_none_or(|f| f.disk_state().is_deleted())
                        {
                            breakpoint_store.remove_breakpoints_of_buffer(entity_id, cx);
                            return;
                        }

                        if let Some(abs_path) = BreakpointStore::abs_path_from_buffer(&buffer, cx) {
                            if breakpoint_store.breakpoints.contains_key(&abs_path) {
                                return;
                            }

                            if let Some(old_path) = breakpoint_store
                                .breakpoints
                                .iter()
                                .find(|(_, in_file)| in_file.buffer.entity_id() == entity_id)
                                .map(|values| values.0)
                                .cloned()
                            {
                                breakpoint_store.on_file_rename(old_path, abs_path, cx);
                            }
                        }
                    }
                    _ => {}
                },
            ));

            BreakpointsInFile {
                buffer,
                breakpoints: Vec::new(),
                _subscription: subscription,
            }
        }

        pub(super) fn merge_restored(
            &mut self,
            restored: BreakpointsInFile,
            cx: &App,
        ) -> Vec<text::Anchor> {
            let source = restored.buffer.read(cx).snapshot();
            let destination = self.buffer.read(cx).snapshot();
            let same_buffer = restored.buffer == self.buffer;
            let max_row = destination.max_point_utf16().row;
            let mut added = Vec::new();
            let live_rows = self
                .breakpoints
                .iter()
                .map(|breakpoint| {
                    destination
                        .summary_for_anchor::<PointUtf16>(breakpoint.position())
                        .row
                })
                .collect::<HashSet<_>>();
            for mut breakpoint in restored.breakpoints {
                let row = source
                    .summary_for_anchor::<PointUtf16>(breakpoint.position())
                    .row;
                if row > max_row || live_rows.contains(&row) {
                    continue;
                }
                if !same_buffer {
                    breakpoint.bp.position = destination.anchor_after(PointUtf16::new(row, 0));
                }
                added.push(*breakpoint.position());
                self.breakpoints.push(breakpoint);
            }
            added
        }
    }
}

#[derive(Clone)]
struct RemoteBreakpointStore {
    upstream_client: AnyProtoClient,
    upstream_project_id: u64,
}

#[derive(Clone)]
enum BreakpointStoreMode {
    Local,
    Remote(RemoteBreakpointStore),
}

#[derive(Clone, PartialEq)]
pub struct ActiveStackFrame {
    pub session_id: SessionId,
    pub thread_id: ThreadId,
    pub stack_frame_id: StackFrameId,
    pub path: Arc<Path>,
    pub position: text::Anchor,
}

pub struct BreakpointStore {
    buffer_store: Entity<BufferStore>,
    worktree_store: Entity<WorktreeStore>,
    breakpoints: BTreeMap<Arc<Path>, BreakpointsInFile>,
    pending_restored: BTreeMap<Arc<Path>, Vec<SourceBreakpoint>>,
    restoring_paths: collections::HashSet<Arc<Path>>,
    pending_restored_buffers: Option<Subscription>,
    pending_restored_buffer_changes: HashMap<Arc<Path>, Subscription>,
    downstream_client: Option<(AnyProtoClient, u64)>,
    active_stack_frame: Option<ActiveStackFrame>,
    active_debug_line_pane_id: Option<EntityId>,
    remote_snapshots_in_flight: HashMap<Arc<Path>, u64>,
    remote_snapshots_received: u64,
    remote_snapshot_versions: HashMap<Arc<Path>, u64>,
    downstream_version: Cell<u64>,
    downstream_path_versions: RefCell<HashMap<Arc<Path>, u64>>,
    collaborators: HashSet<proto::PeerId>,
    remote_edits: HashMap<RemoteEditSender, VecDeque<RemoteBreakpointEdit>>,
    remote_edit_buffers: HashMap<EntityId, Subscription>,
    upstream_requests: Option<(mpsc::UnboundedSender<VecDeque<UpstreamEditId>>, Task<()>)>,
    pending_upstream_edits: VecDeque<UpstreamEdit>,
    next_upstream_edit_id: UpstreamEditId,
    confirmed_remote_breakpoints: HashMap<Arc<Path>, ConfirmedBreakpoints>,
    applied_remote_snapshot_versions: HashMap<Arc<Path>, u64>,
    // E.g ssh
    mode: BreakpointStoreMode,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
struct UpstreamEditId(u64);

struct UpstreamEdit {
    id: UpstreamEditId,
    path: Arc<Path>,
    buffer: Entity<Buffer>,
    breakpoint: BreakpointWithPosition,
    edit: proto::BreakpointEdit,
    confirmed_by: Option<u64>,
}

#[derive(Default)]
struct ConfirmedBreakpoints {
    buffer: Option<Entity<Buffer>>,
    breakpoints: Vec<StatefulBreakpoint>,
    version: Option<u64>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum RemoteEditSender {
    Collaborator(proto::PeerId),
    Direct(proto::PeerId),
}

struct RemoteBreakpointEdit {
    buffer: Entity<Buffer>,
    breakpoints: Vec<(BreakpointWithPosition, bool)>,
    edit: proto::BreakpointEdit,
    applied: oneshot::Sender<Result<()>>,
}

impl BreakpointStore {
    pub fn init(client: &AnyProtoClient) {
        client.add_entity_request_handler(Self::handle_toggle_breakpoint);
        client.add_entity_message_handler(Self::handle_breakpoints_for_file);
    }
    pub fn local(worktree_store: Entity<WorktreeStore>, buffer_store: Entity<BufferStore>) -> Self {
        BreakpointStore {
            breakpoints: BTreeMap::new(),
            pending_restored: BTreeMap::new(),
            restoring_paths: collections::HashSet::default(),
            pending_restored_buffers: None,
            pending_restored_buffer_changes: HashMap::default(),
            mode: BreakpointStoreMode::Local,
            buffer_store,
            worktree_store,
            downstream_client: None,
            active_stack_frame: Default::default(),
            active_debug_line_pane_id: None,
            remote_snapshots_in_flight: HashMap::default(),
            remote_snapshots_received: 0,
            remote_snapshot_versions: HashMap::default(),
            downstream_version: Cell::new(0),
            downstream_path_versions: RefCell::default(),
            collaborators: HashSet::default(),
            remote_edits: HashMap::default(),
            remote_edit_buffers: HashMap::default(),
            upstream_requests: None,
            pending_upstream_edits: VecDeque::new(),
            next_upstream_edit_id: UpstreamEditId::default(),
            confirmed_remote_breakpoints: HashMap::default(),
            applied_remote_snapshot_versions: HashMap::default(),
        }
    }

    pub(crate) fn remote(
        upstream_project_id: u64,
        upstream_client: AnyProtoClient,
        buffer_store: Entity<BufferStore>,
        worktree_store: Entity<WorktreeStore>,
    ) -> Self {
        BreakpointStore {
            breakpoints: BTreeMap::new(),
            pending_restored: BTreeMap::new(),
            restoring_paths: collections::HashSet::default(),
            pending_restored_buffers: None,
            pending_restored_buffer_changes: HashMap::default(),
            mode: BreakpointStoreMode::Remote(RemoteBreakpointStore {
                upstream_client,
                upstream_project_id,
            }),
            buffer_store,
            worktree_store,
            downstream_client: None,
            active_stack_frame: Default::default(),
            active_debug_line_pane_id: None,
            remote_snapshots_in_flight: HashMap::default(),
            remote_snapshots_received: 0,
            remote_snapshot_versions: HashMap::default(),
            downstream_version: Cell::new(0),
            downstream_path_versions: RefCell::default(),
            collaborators: HashSet::default(),
            remote_edits: HashMap::default(),
            remote_edit_buffers: HashMap::default(),
            upstream_requests: None,
            pending_upstream_edits: VecDeque::new(),
            next_upstream_edit_id: UpstreamEditId::default(),
            confirmed_remote_breakpoints: HashMap::default(),
            applied_remote_snapshot_versions: HashMap::default(),
        }
    }

    pub fn shared(&mut self, project_id: u64, downstream_client: AnyProtoClient) {
        self.downstream_client = Some((downstream_client, project_id));
    }

    pub(crate) fn unshared(&mut self, cx: &mut Context<Self>) {
        self.downstream_client.take();
        self.collaborators.clear();
        self.remote_edits.clear();
        self.remote_edit_buffers.clear();

        cx.notify();
    }

    pub(crate) fn set_collaborators(
        &mut self,
        collaborators: impl IntoIterator<Item = proto::PeerId>,
    ) {
        self.collaborators = collaborators.into_iter().collect();
        let collaborators = &self.collaborators;
        self.remote_edits.retain(|sender, _| match sender {
            RemoteEditSender::Collaborator(peer_id) => collaborators.contains(peer_id),
            RemoteEditSender::Direct(_) => true,
        });
        self.retain_remote_edit_buffers();
    }

    pub(crate) fn collaborator_added(&mut self, peer_id: proto::PeerId) {
        self.collaborators.insert(peer_id);
    }

    pub(crate) fn collaborator_removed(&mut self, peer_id: proto::PeerId) {
        self.collaborators.remove(&peer_id);
        self.remote_edits
            .remove(&RemoteEditSender::Collaborator(peer_id));
        self.retain_remote_edit_buffers();
    }

    pub(crate) fn collaborator_renamed(
        &mut self,
        old_peer_id: proto::PeerId,
        new_peer_id: proto::PeerId,
    ) {
        if self.collaborators.remove(&old_peer_id) {
            self.collaborators.insert(new_peer_id);
        }
        if let Some(edits) = self
            .remote_edits
            .remove(&RemoteEditSender::Collaborator(old_peer_id))
        {
            self.remote_edits
                .insert(RemoteEditSender::Collaborator(new_peer_id), edits);
        }
    }

    async fn handle_breakpoints_for_file(
        this: Entity<Self>,
        message: TypedEnvelope<proto::BreakpointsForFile>,
        mut cx: AsyncApp,
    ) -> Result<()> {
        let path = Arc::<Path>::from(Path::new(&message.payload.path));
        let version = message.payload.version;
        let snapshot = this.update(&mut cx, |this, _| {
            if version != 0 {
                if this
                    .remote_snapshot_versions
                    .get(&path)
                    .is_some_and(|latest_version| *latest_version >= version)
                {
                    return None;
                }
                this.remote_snapshot_versions.insert(path.clone(), version);
            }
            this.remote_snapshots_received += 1;
            this.remote_snapshots_in_flight
                .insert(path.clone(), this.remote_snapshots_received);
            Some(this.remote_snapshots_received)
        });
        let Some(snapshot) = snapshot else {
            return Ok(());
        };
        if message.payload.breakpoints.is_empty() {
            this.update(&mut cx, |this, cx| {
                this.finish_remote_snapshot(&path, snapshot);
                if this.apply_remote_snapshot(&path, Vec::new(), None, version, cx) {
                    this.send_breakpoints_downstream(&path);
                    cx.notify();
                }
            });
            return Ok(());
        }

        let breakpoints = message
            .payload
            .breakpoints
            .into_iter()
            .filter_map(|breakpoint| {
                let has_id = breakpoint.id.is_some();
                let session_state = breakpoint
                    .session_state
                    .iter()
                    .map(|(session_id, state)| {
                        let state = BreakpointSessionState {
                            id: state.id,
                            verified: state.verified,
                        };
                        (SessionId::from_proto(*session_id), state)
                    })
                    .collect();
                let bp = BreakpointWithPosition::from_proto(breakpoint).log_err()?;
                Some((StatefulBreakpoint { bp, session_state }, has_id))
            })
            .collect::<Vec<_>>();

        let open_buffer = this.update(&mut cx, |this, cx| {
            let Some(project_path) = this
                .worktree_store
                .read(cx)
                .project_path_for_absolute_path(message.payload.path.as_ref(), cx)
            else {
                this.finish_remote_snapshot(&path, snapshot);
                return None;
            };
            Some(this.buffer_store.update(cx, |buffer_store, cx| {
                buffer_store.open_buffer(project_path, cx)
            }))
        });
        let open_buffer = open_buffer.context("Invalid project path")?;
        let buffer = async {
            let buffer = open_buffer.await?;
            Self::wait_for_remote_anchors(
                &buffer,
                breakpoints
                    .iter()
                    .map(|(breakpoint, _)| *breakpoint.position()),
                &mut cx,
            )
            .await?;
            anyhow::Ok(buffer)
        }
        .await;

        this.update(&mut cx, move |this, cx| {
            if !this.finish_remote_snapshot(&path, snapshot) {
                return anyhow::Ok(());
            }
            let buffer = buffer?;
            let buffer_snapshot = buffer.read(cx).snapshot();
            let breakpoints = breakpoints
                .into_iter()
                .filter(|(breakpoint, _)| buffer_snapshot.can_resolve(breakpoint.position()))
                .collect::<Vec<_>>();
            let breakpoints = this.adopt_known_identities(&path, &buffer_snapshot, breakpoints);
            this.apply_remote_snapshot(&path, breakpoints, Some(buffer), version, cx);
            this.send_breakpoints_downstream(&path);
            cx.notify();
            anyhow::Ok(())
        })
    }

    fn apply_remote_snapshot(
        &mut self,
        path: &Arc<Path>,
        breakpoints: Vec<StatefulBreakpoint>,
        buffer: Option<Entity<Buffer>>,
        version: u64,
        cx: &mut Context<Self>,
    ) -> bool {
        if version != 0 {
            let applied = self
                .applied_remote_snapshot_versions
                .entry(path.clone())
                .or_default();
            *applied = (*applied).max(version);
        }
        let applied_version = self.applied_remote_snapshot_versions.get(path).copied();
        self.pending_upstream_edits.retain(|edit| {
            edit.path != *path
                || edit.confirmed_by.is_none_or(|confirmed_by| {
                    version != 0 && applied_version.is_none_or(|applied| applied < confirmed_by)
                })
        });
        self.confirmed_remote_breakpoints.insert(
            path.clone(),
            ConfirmedBreakpoints {
                buffer: buffer.clone(),
                breakpoints,
                version: Some(version),
            },
        );
        self.apply_confirmed_remote_breakpoints(path, buffer, cx)
    }

    fn apply_confirmed_remote_breakpoints(
        &mut self,
        path: &Arc<Path>,
        buffer: Option<Entity<Buffer>>,
        cx: &mut Context<Self>,
    ) -> bool {
        let Some(confirmed) = self.confirmed_remote_breakpoints.get(path) else {
            return false;
        };
        let confirmed_buffer = confirmed.buffer.clone();
        let confirmed_breakpoints = confirmed.breakpoints.clone();
        let file_breakpoints = self.breakpoints.remove(path);
        let previous = file_breakpoints
            .as_ref()
            .map(|file_breakpoints| breakpoint_positions(&file_breakpoints.breakpoints))
            .unwrap_or_default();
        let pending_edits = self
            .pending_upstream_edits
            .iter()
            .filter(|edit| confirmed.version != Some(0) && edit.path == *path)
            .collect::<Vec<_>>();
        let target_buffer = buffer
            .or_else(|| {
                file_breakpoints
                    .as_ref()
                    .map(|file_breakpoints| file_breakpoints.buffer.clone())
            })
            .or_else(|| self.open_buffer_for_abs_path(path, cx))
            .or_else(|| confirmed_buffer.clone())
            .or_else(|| pending_edits.last().map(|edit| edit.buffer.clone()));
        if let Some(target_buffer) = target_buffer {
            let destination = target_buffer.read(cx).snapshot();
            let mut breakpoints = Vec::new();
            if let Some(confirmed_buffer) = &confirmed_buffer {
                let source = confirmed_buffer.read(cx).snapshot();
                breakpoints.extend(confirmed_breakpoints.into_iter().filter_map(|breakpoint| {
                    Some(StatefulBreakpoint {
                        bp: reanchor_breakpoint(breakpoint.bp, &source, &destination)?,
                        session_state: breakpoint.session_state,
                    })
                }));
            }
            for edit in pending_edits {
                let source = edit.buffer.read(cx).snapshot();
                if let Some(breakpoint) =
                    reanchor_breakpoint(edit.breakpoint.clone(), &source, &destination)
                {
                    apply_upstream_edit_to(&mut breakpoints, breakpoint, edit.edit);
                }
            }
            if !breakpoints.is_empty() {
                let mut file_breakpoints = file_breakpoints
                    .filter(|file_breakpoints| file_breakpoints.buffer == target_buffer)
                    .unwrap_or_else(|| BreakpointsInFile::new(target_buffer, cx));
                file_breakpoints.breakpoints = breakpoints;
                self.breakpoints.insert(path.clone(), file_breakpoints);
            }
        }
        let current = self
            .breakpoints
            .get(path)
            .map(|file_breakpoints| breakpoint_positions(&file_breakpoints.breakpoints))
            .unwrap_or_default();
        previous != current
    }

    fn open_buffer_for_abs_path(&self, path: &Path, cx: &App) -> Option<Entity<Buffer>> {
        let project_path = self
            .worktree_store
            .read(cx)
            .project_path_for_absolute_path(path, cx)?;
        self.buffer_store.read(cx).get_by_path(&project_path)
    }

    fn enqueue_upstream_edits(
        &mut self,
        path: Arc<Path>,
        buffer: Entity<Buffer>,
        edits: Vec<(BreakpointWithPosition, proto::BreakpointEdit)>,
        cx: &mut Context<Self>,
    ) {
        let BreakpointStoreMode::Remote(remote) = &self.mode else {
            return;
        };
        let upstream_client = remote.upstream_client.clone();
        let project_id = remote.upstream_project_id;
        let edit_ids = edits
            .into_iter()
            .map(|(breakpoint, edit)| {
                self.push_upstream_edit(path.clone(), buffer.clone(), breakpoint, edit)
            })
            .collect::<VecDeque<_>>();
        if edit_ids.is_empty() {
            return;
        }
        let (upstream_requests, _) = self.upstream_requests.get_or_insert_with(|| {
            let (upstream_requests, mut queued_edits) =
                mpsc::unbounded::<VecDeque<UpstreamEditId>>();
            let task = cx.spawn(async move |breakpoint_store, cx| {
                let mut batch_supported = true;
                while let Some(mut edit_ids) = queued_edits.next().await {
                    while !edit_ids.is_empty() {
                        let request = breakpoint_store.update(cx, |breakpoint_store, cx| {
                            let mut request: Option<proto::ToggleBreakpoint> = None;
                            let mut sent_ids = Vec::new();
                            while let Some(edit_id) = edit_ids.front().copied() {
                                let Some(next) =
                                    breakpoint_store.upstream_request(edit_id, project_id, cx)
                                else {
                                    edit_ids.pop_front();
                                    continue;
                                };
                                if let Some(request) = &mut request {
                                    if !batch_supported
                                        || request.path != next.path
                                        || request.edit != next.edit
                                        || !matches!(
                                            request.edit(),
                                            proto::BreakpointEdit::Set
                                                | proto::BreakpointEdit::Remove
                                        )
                                    {
                                        break;
                                    }
                                    request.additional_breakpoints.extend(next.breakpoint);
                                } else {
                                    request = Some(next);
                                }
                                edit_ids.pop_front();
                                sent_ids.push(edit_id);
                            }
                            request.map(|request| (request, sent_ids))
                        });
                        let (request, mut sent_ids) = match request {
                            Ok(Some(request)) => request,
                            Ok(None) => continue,
                            Err(_) => return,
                        };
                        let response = upstream_client
                            .request(request)
                            .await
                            .and_then(|response| {
                                anyhow::ensure!(
                                    response.processed_breakpoints <= sent_ids.len() as u64,
                                    "Host acknowledged more breakpoint edits than requested"
                                );
                                Ok(response)
                            })
                            .log_err();
                        let processed = response.as_ref().map_or(sent_ids.len(), |response| {
                            (response.processed_breakpoints as usize).max(1)
                        });
                        if processed < sent_ids.len() {
                            batch_supported = false;
                            for edit_id in sent_ids.drain(processed..).rev() {
                                edit_ids.push_front(edit_id);
                            }
                        }
                        if breakpoint_store
                            .update(cx, |breakpoint_store, cx| {
                                breakpoint_store.finish_upstream_edits(
                                    &sent_ids,
                                    response.map(|response| response.version),
                                    cx,
                                )
                            })
                            .is_err()
                        {
                            return;
                        }
                    }
                }
            });
            (upstream_requests, task)
        });
        upstream_requests.unbounded_send(edit_ids).log_err();
    }

    fn push_upstream_edit(
        &mut self,
        path: Arc<Path>,
        buffer: Entity<Buffer>,
        breakpoint: BreakpointWithPosition,
        edit: proto::BreakpointEdit,
    ) -> UpstreamEditId {
        let id = self.next_upstream_edit_id;
        self.next_upstream_edit_id = UpstreamEditId(id.0 + 1);
        self.pending_upstream_edits.push_back(UpstreamEdit {
            id,
            path,
            buffer,
            breakpoint,
            edit,
            confirmed_by: None,
        });
        id
    }

    fn upstream_request(
        &mut self,
        edit_id: UpstreamEditId,
        project_id: u64,
        cx: &App,
    ) -> Option<proto::ToggleBreakpoint> {
        let edit = self
            .pending_upstream_edits
            .iter_mut()
            .find(|edit| edit.id == edit_id)?;
        if let Some(current_path) = Self::abs_path_from_buffer(&edit.buffer, cx) {
            edit.path = current_path;
        }
        let path = edit.path.clone();
        let current_buffer = self
            .breakpoints
            .get(&path)
            .map(|file_breakpoints| file_breakpoints.buffer.clone())
            .or_else(|| self.open_buffer_for_abs_path(&path, cx));
        let edit = self
            .pending_upstream_edits
            .iter_mut()
            .find(|edit| edit.id == edit_id)?;
        if let Some(current_buffer) = current_buffer
            && current_buffer != edit.buffer
            && let Some(breakpoint) = reanchor_breakpoint(
                edit.breakpoint.clone(),
                &edit.buffer.read(cx).snapshot(),
                &current_buffer.read(cx).snapshot(),
            )
        {
            edit.breakpoint = breakpoint;
            edit.buffer = current_buffer;
        }
        Some(proto::ToggleBreakpoint {
            project_id,
            path: edit.path.to_string_lossy().into_owned(),
            breakpoint: Some(edit.breakpoint.to_proto(&HashMap::default())),
            edit: i32::from(edit.edit),
            additional_breakpoints: Vec::new(),
        })
    }

    fn finish_upstream_edits(
        &mut self,
        edit_ids: &[UpstreamEditId],
        response_version: Option<u64>,
        cx: &mut Context<Self>,
    ) {
        let edit_ids = edit_ids.iter().copied().collect::<HashSet<_>>();
        let mut changed_paths = HashSet::default();
        self.pending_upstream_edits.retain_mut(|edit| {
            if !edit_ids.contains(&edit.id) {
                return true;
            }
            if response_version == Some(0) {
                return false;
            }
            if let Some(version) = response_version {
                let applied_version = self
                    .applied_remote_snapshot_versions
                    .get(&edit.path)
                    .copied();
                if applied_version.is_none_or(|applied| applied < version) {
                    edit.confirmed_by = Some(version);
                    return true;
                }
            }
            changed_paths.insert(edit.path.clone());
            false
        });
        for path in changed_paths {
            self.confirmed_remote_breakpoints
                .entry(path.clone())
                .or_default();
            if self.apply_confirmed_remote_breakpoints(&path, None, cx) {
                self.send_breakpoints_downstream(&path);
                cx.emit(BreakpointStoreEvent::BreakpointsUpdated(
                    path,
                    BreakpointUpdatedReason::Toggled,
                ));
                cx.notify();
            }
        }
    }

    fn adopt_known_identities(
        &self,
        path: &Arc<Path>,
        snapshot: &BufferSnapshot,
        incoming: Vec<(StatefulBreakpoint, bool)>,
    ) -> Vec<StatefulBreakpoint> {
        let row = |breakpoint: &BreakpointWithPosition| {
            snapshot
                .summary_for_anchor::<PointUtf16>(&breakpoint.position)
                .row
        };
        let mut known = Vec::<&BreakpointWithPosition>::new();
        let displayed = self
            .breakpoints
            .get(path)
            .into_iter()
            .flat_map(|file_breakpoints| &file_breakpoints.breakpoints)
            .map(|breakpoint| &breakpoint.bp);
        let pending = self
            .pending_upstream_edits
            .iter()
            .filter(|edit| edit.path == *path)
            .map(|edit| &edit.breakpoint);
        for candidate in displayed.chain(pending) {
            let identified_by_snapshot = incoming
                .iter()
                .any(|(breakpoint, has_id)| *has_id && breakpoint.bp.id == candidate.id);
            if snapshot.can_resolve(&candidate.position)
                && !identified_by_snapshot
                && known.iter().all(|known| known.id != candidate.id)
            {
                known.push(candidate);
            }
        }

        let mut adopted = vec![None; incoming.len()];
        for require_equal_value in [true, false] {
            for ((breakpoint, has_id), adopted) in incoming.iter().zip(&mut adopted) {
                if *has_id || adopted.is_some() {
                    continue;
                }
                let incoming_row = row(&breakpoint.bp);
                if let Some(index) = known.iter().position(|known| {
                    row(known) == incoming_row
                        && (!require_equal_value || known.bp == breakpoint.bp.bp)
                }) {
                    *adopted = Some(known.remove(index).id);
                }
            }
        }
        incoming
            .into_iter()
            .zip(adopted)
            .map(|((mut breakpoint, _), adopted)| {
                if let Some(id) = adopted {
                    breakpoint.bp.id = id;
                }
                breakpoint
            })
            .collect()
    }

    async fn wait_for_remote_anchors(
        buffer: &Entity<Buffer>,
        anchors: impl IntoIterator<Item = text::Anchor>,
        cx: &mut AsyncApp,
    ) -> Result<()> {
        buffer
            .update(cx, |buffer, _| {
                let buffer_id = buffer.remote_id();
                let anchors = anchors
                    .into_iter()
                    .filter(|anchor| anchor.buffer_id == buffer_id)
                    .collect::<Vec<_>>();
                buffer.wait_for_anchors(anchors)
            })
            .await
    }

    fn enqueue_remote_edit(
        &mut self,
        sender: RemoteEditSender,
        edit: RemoteBreakpointEdit,
        cx: &mut Context<Self>,
    ) -> Result<()> {
        if let RemoteEditSender::Collaborator(peer_id) = sender {
            anyhow::ensure!(
                self.collaborators.contains(&peer_id),
                "Peer {peer_id} is not a collaborator of this project"
            );
        }
        self.remote_edit_buffers
            .entry(edit.buffer.entity_id())
            .or_insert_with(|| {
                cx.observe(&edit.buffer, |breakpoint_store, _, cx| {
                    breakpoint_store.apply_remote_edits(cx)
                })
            });
        self.remote_edits.entry(sender).or_default().push_back(edit);
        self.apply_remote_edits(cx);
        Ok(())
    }

    fn apply_remote_edits(&mut self, cx: &mut Context<Self>) {
        let senders = self.remote_edits.keys().copied().collect::<Vec<_>>();
        for sender in senders {
            while let Some((edit, readiness)) = self.pop_ready_remote_edit(sender, cx) {
                let RemoteBreakpointEdit {
                    buffer,
                    breakpoints,
                    edit,
                    applied,
                } = edit;
                let result = readiness.map(|()| match edit {
                    proto::BreakpointEdit::Toggle => {
                        if let Some((breakpoint, _)) = breakpoints.into_iter().next() {
                            self.toggle_breakpoint(
                                buffer,
                                breakpoint,
                                BreakpointEditAction::Toggle,
                                cx,
                            );
                        }
                    }
                    proto::BreakpointEdit::Set => {
                        self.apply_upstream_edit(buffer, breakpoints, true, cx)
                    }
                    proto::BreakpointEdit::Remove => {
                        self.apply_upstream_edit(buffer, breakpoints, false, cx)
                    }
                    proto::BreakpointEdit::Superseded => {}
                });
                applied.send(result).ok();
            }
        }
        self.retain_remote_edit_buffers();
    }

    fn retain_remote_edit_buffers(&mut self) {
        let remote_edits = &self.remote_edits;
        self.remote_edit_buffers.retain(|buffer_id, _| {
            remote_edits
                .values()
                .flatten()
                .any(|edit| edit.buffer.entity_id() == *buffer_id)
        });
    }

    fn pop_ready_remote_edit(
        &mut self,
        sender: RemoteEditSender,
        cx: &App,
    ) -> Option<(RemoteBreakpointEdit, Result<()>)> {
        let edits = self.remote_edits.get_mut(&sender)?;
        let readiness = match edits.front() {
            Some(edit) => 'readiness: {
                let buffer = edit.buffer.read(cx);
                let mut pending = false;
                for (breakpoint, _) in &edit.breakpoints {
                    let position = breakpoint.position;
                    if position.buffer_id != buffer.remote_id() {
                        break 'readiness Err(anyhow!(
                            "Breakpoint position {position:?} belongs to another buffer"
                        ));
                    }
                    if !buffer.can_resolve(&position) {
                        if position.timestamp().replica_id == buffer.replica_id() {
                            break 'readiness Err(anyhow!(
                                "Breakpoint position {position:?} refers to an edit this replica never made"
                            ));
                        }
                        pending = true;
                    }
                }
                if pending {
                    return None;
                }
                Ok(())
            }
            None => {
                self.remote_edits.remove(&sender);
                return None;
            }
        };
        let edit = edits.pop_front()?;
        if edits.is_empty() {
            self.remote_edits.remove(&sender);
        }
        Some((edit, readiness))
    }

    fn finish_remote_snapshot(&mut self, path: &Arc<Path>, snapshot: u64) -> bool {
        let latest = self.remote_snapshots_in_flight.get(path) == Some(&snapshot);
        if latest {
            self.remote_snapshots_in_flight.remove(path);
        }
        latest
    }

    async fn handle_toggle_breakpoint(
        this: Entity<Self>,
        message: TypedEnvelope<proto::ToggleBreakpoint>,
        mut cx: AsyncApp,
    ) -> Result<proto::ToggleBreakpointResponse> {
        let sender = match message.original_sender_id {
            Some(peer_id) => RemoteEditSender::Collaborator(peer_id),
            None => RemoteEditSender::Direct(message.sender_id),
        };
        let path = this
            .update(&mut cx, |this, cx| {
                this.worktree_store
                    .read(cx)
                    .project_path_for_absolute_path(message.payload.path.as_ref(), cx)
            })
            .context("Could not resolve provided abs path")?;
        let buffer = this
            .update(&mut cx, |this, cx| {
                this.buffer_store.read(cx).get_by_path(&path)
            })
            .context("Could not find buffer for a given path")?;
        let edit = message.payload.edit();
        anyhow::ensure!(
            message.payload.additional_breakpoints.is_empty()
                || matches!(
                    edit,
                    proto::BreakpointEdit::Set | proto::BreakpointEdit::Remove
                ),
            "Only explicit breakpoint additions and removals can be batched"
        );
        let breakpoint = message
            .payload
            .breakpoint
            .context("Breakpoint not present in RPC payload")?;
        let breakpoints = std::iter::once(breakpoint)
            .chain(message.payload.additional_breakpoints)
            .map(|breakpoint| {
                let has_id = breakpoint.id.is_some();
                Ok((BreakpointWithPosition::from_proto(breakpoint)?, has_id))
            })
            .collect::<Result<Vec<_>>>()?;
        let processed_breakpoints = if edit == proto::BreakpointEdit::Toggle {
            0
        } else {
            breakpoints.len() as u64
        };

        let (applied, edit_applied) = oneshot::channel();
        this.update(&mut cx, |this, cx| {
            this.enqueue_remote_edit(
                sender,
                RemoteBreakpointEdit {
                    buffer: buffer.clone(),
                    breakpoints,
                    edit,
                    applied,
                },
                cx,
            )
        })?;
        edit_applied
            .await
            .context("Breakpoint edit was cancelled")??;
        let version = if edit == proto::BreakpointEdit::Toggle {
            0
        } else {
            this.update(&mut cx, |breakpoint_store, cx| {
                Self::abs_path_from_buffer(&buffer, cx)
                    .and_then(|path| {
                        breakpoint_store
                            .downstream_path_versions
                            .borrow()
                            .get(&path)
                            .copied()
                    })
                    .unwrap_or(0)
            })
        };
        Ok(proto::ToggleBreakpointResponse {
            version,
            processed_breakpoints,
        })
    }

    pub(crate) fn broadcast(&self) {
        for path in self.breakpoints.keys() {
            self.send_breakpoints_downstream(path);
        }
    }

    pub(crate) fn update_session_breakpoint(
        &mut self,
        session_id: SessionId,
        _: dap::BreakpointEventReason,
        breakpoint: dap::Breakpoint,
    ) {
        maybe!({
            let event_id = breakpoint.id?;

            let state = self
                .breakpoints
                .values_mut()
                .find_map(|breakpoints_in_file| {
                    breakpoints_in_file
                        .breakpoints
                        .iter_mut()
                        .find_map(|state| {
                            let state = state.session_state.get_mut(&session_id)?;

                            if state.id == event_id {
                                Some(state)
                            } else {
                                None
                            }
                        })
                })?;

            state.verified = breakpoint.verified;
            Some(())
        });
    }

    pub(super) fn mark_breakpoints_verified(
        &mut self,
        session_id: SessionId,
        abs_path: &Path,

        it: impl Iterator<Item = (BreakpointWithPosition, BreakpointSessionState)>,
    ) {
        maybe!({
            let breakpoints = self.breakpoints.get_mut(abs_path)?;
            for (breakpoint, state) in it {
                if let Some(to_update) = breakpoints
                    .breakpoints
                    .iter_mut()
                    .find(|existing| existing.bp.id == breakpoint.id)
                {
                    to_update
                        .session_state
                        .entry(session_id)
                        .insert_entry(state);
                }
            }
            Some(())
        });
    }

    pub fn abs_path_from_buffer(buffer: &Entity<Buffer>, cx: &App) -> Option<Arc<Path>> {
        worktree::File::from_dyn(buffer.read(cx).file())
            .map(|file| file.worktree.read(cx).absolutize(&file.path))
            .map(Arc::<Path>::from)
    }

    pub fn toggle_breakpoint(
        &mut self,
        buffer: Entity<Buffer>,
        mut breakpoint: BreakpointWithPosition,
        edit_action: BreakpointEditAction,
        cx: &mut Context<Self>,
    ) {
        let Some(abs_path) = Self::abs_path_from_buffer(&buffer, cx) else {
            return;
        };
        let row = buffer
            .read(cx)
            .summary_for_anchor::<PointUtf16>(&breakpoint.position)
            .row;
        self.forget_pending_restored(&abs_path, row);

        let breakpoint_set = self.file_breakpoints_for_buffer(&abs_path, buffer.clone(), cx);
        let previous_breakpoints = breakpoint_positions(&breakpoint_set.breakpoints);
        let existing = breakpoint_index(&breakpoint_set.breakpoints, &breakpoint);

        match edit_action {
            BreakpointEditAction::Toggle => {
                if let Some(index) = existing {
                    breakpoint_set.breakpoints.remove(index);
                } else {
                    // We did not remove any breakpoint, hence let's toggle one.
                    breakpoint_set
                        .breakpoints
                        .push(StatefulBreakpoint::new(breakpoint));
                }
            }
            BreakpointEditAction::InvertState => {
                if let Some(bp) =
                    existing.and_then(|index| breakpoint_set.breakpoints.get_mut(index))
                {
                    let bp = &mut bp.bp.bp;
                    if bp.is_enabled() {
                        bp.state = BreakpointState::Disabled;
                    } else {
                        bp.state = BreakpointState::Enabled;
                    }
                } else {
                    breakpoint.bp.state = BreakpointState::Disabled;
                    breakpoint_set
                        .breakpoints
                        .push(StatefulBreakpoint::new(breakpoint));
                }
            }
            BreakpointEditAction::EditLogMessage(log_message) => {
                if !log_message.is_empty() {
                    let found_bp = breakpoint_to_edit(&mut breakpoint_set.breakpoints, existing);

                    if let Some(found_bp) = found_bp {
                        found_bp.message = Some(log_message);
                    } else {
                        breakpoint.bp.message = Some(log_message);
                        // We did not remove any breakpoint, hence let's toggle one.
                        breakpoint_set
                            .breakpoints
                            .push(StatefulBreakpoint::new(breakpoint));
                    }
                } else if breakpoint.bp.message.is_some() {
                    if let Some(position) = existing {
                        breakpoint_set.breakpoints.remove(position);
                    } else {
                        log::error!("Failed to find position of breakpoint to delete")
                    }
                }
            }
            BreakpointEditAction::EditHitCondition(hit_condition) => {
                if !hit_condition.is_empty() {
                    let found_bp = breakpoint_to_edit(&mut breakpoint_set.breakpoints, existing);

                    if let Some(found_bp) = found_bp {
                        found_bp.hit_condition = Some(hit_condition);
                    } else {
                        breakpoint.bp.hit_condition = Some(hit_condition);
                        // We did not remove any breakpoint, hence let's toggle one.
                        breakpoint_set
                            .breakpoints
                            .push(StatefulBreakpoint::new(breakpoint))
                    }
                } else if breakpoint.bp.hit_condition.is_some() {
                    if let Some(position) = existing {
                        breakpoint_set.breakpoints.remove(position);
                    } else {
                        log::error!("Failed to find position of breakpoint to delete")
                    }
                }
            }
            BreakpointEditAction::EditCondition(condition) => {
                if !condition.is_empty() {
                    let found_bp = breakpoint_to_edit(&mut breakpoint_set.breakpoints, existing);

                    if let Some(found_bp) = found_bp {
                        found_bp.condition = Some(condition);
                    } else {
                        breakpoint.bp.condition = Some(condition);
                        // We did not remove any breakpoint, hence let's toggle one.
                        breakpoint_set
                            .breakpoints
                            .push(StatefulBreakpoint::new(breakpoint));
                    }
                } else if breakpoint.bp.condition.is_some() {
                    if let Some(position) = existing {
                        breakpoint_set.breakpoints.remove(position);
                    } else {
                        log::error!("Failed to find position of breakpoint to delete")
                    }
                }
            }
        }

        if breakpoint_set.breakpoints.is_empty() {
            self.breakpoints.remove(&abs_path);
        }
        self.sync_file_breakpoints(&abs_path, &buffer, &previous_breakpoints, cx);

        cx.emit(BreakpointStoreEvent::BreakpointsUpdated(
            abs_path,
            BreakpointUpdatedReason::Toggled,
        ));
        cx.notify();
    }

    fn file_breakpoints_for_buffer(
        &mut self,
        abs_path: &Arc<Path>,
        buffer: Entity<Buffer>,
        cx: &mut Context<Self>,
    ) -> &mut BreakpointsInFile {
        let breakpoint_set = self
            .breakpoints
            .entry(abs_path.clone())
            .or_insert_with(|| BreakpointsInFile::new(buffer.clone(), cx));

        // Buffers changed for the file, migrate breakpoints to the new buffer
        if breakpoint_set.buffer != buffer {
            let old_snapshot = breakpoint_set.buffer.read(cx).snapshot();
            let new_snapshot = buffer.read(cx).snapshot();
            let breakpoints = breakpoint_set
                .breakpoints
                .drain(..)
                .filter_map(|mut breakpoint| {
                    breakpoint.bp =
                        reanchor_breakpoint(breakpoint.bp, &old_snapshot, &new_snapshot)?;
                    Some(breakpoint)
                })
                .collect();
            *breakpoint_set = BreakpointsInFile::new(buffer, cx);
            breakpoint_set.breakpoints = breakpoints;
        }
        breakpoint_set
    }

    fn apply_upstream_edit(
        &mut self,
        buffer: Entity<Buffer>,
        breakpoints: Vec<(BreakpointWithPosition, bool)>,
        keep: bool,
        cx: &mut Context<Self>,
    ) {
        let Some(abs_path) = Self::abs_path_from_buffer(&buffer, cx) else {
            return;
        };
        for (breakpoint, _) in &breakpoints {
            let row = buffer
                .read(cx)
                .summary_for_anchor::<PointUtf16>(&breakpoint.position)
                .row;
            self.forget_pending_restored(&abs_path, row);
        }

        let breakpoint_set = self.file_breakpoints_for_buffer(&abs_path, buffer.clone(), cx);
        let previous_breakpoints = breakpoint_positions(&breakpoint_set.breakpoints);
        for (breakpoint, has_id) in breakpoints {
            let existing = if has_id {
                breakpoint_set
                    .breakpoints
                    .iter()
                    .position(|existing| existing.bp.id == breakpoint.id)
            } else {
                breakpoint_index(&breakpoint_set.breakpoints, &breakpoint)
            };
            match (existing, keep) {
                (Some(index), true) => {
                    if let Some(existing) = breakpoint_set.breakpoints.get_mut(index) {
                        existing.bp = BreakpointWithPosition {
                            id: existing.bp.id,
                            ..breakpoint
                        };
                    }
                }
                (None, true) => breakpoint_set
                    .breakpoints
                    .push(StatefulBreakpoint::new(breakpoint)),
                (Some(index), false) => {
                    breakpoint_set.breakpoints.remove(index);
                }
                (None, false) => {}
            }
        }
        let changed = breakpoint_set
            .breakpoints
            .iter()
            .map(|breakpoint| &breakpoint.bp)
            .ne(previous_breakpoints.iter());
        if breakpoint_set.breakpoints.is_empty() {
            self.breakpoints.remove(&abs_path);
        }
        self.sync_file_breakpoints(&abs_path, &buffer, &previous_breakpoints, cx);
        if !changed {
            return;
        }
        cx.emit(BreakpointStoreEvent::BreakpointsUpdated(
            abs_path,
            BreakpointUpdatedReason::Toggled,
        ));
        cx.notify();
    }

    fn sync_file_breakpoints(
        &mut self,
        abs_path: &Arc<Path>,
        buffer: &Entity<Buffer>,
        previous_breakpoints: &[BreakpointWithPosition],
        cx: &mut Context<Self>,
    ) {
        if let BreakpointStoreMode::Remote(_) = &self.mode {
            let current_breakpoints = self
                .breakpoints
                .get(abs_path)
                .map(|file_breakpoints| file_breakpoints.breakpoints.as_slice())
                .unwrap_or_default();
            let edits = upstream_edits(previous_breakpoints, current_breakpoints)
                .into_iter()
                .map(|(breakpoint, edit)| (breakpoint.clone(), edit))
                .collect::<Vec<_>>();
            self.enqueue_upstream_edits(abs_path.clone(), buffer.clone(), edits, cx);
        }
        self.send_breakpoints_downstream(abs_path);
    }

    fn remove_breakpoints_of_buffer(&mut self, buffer_id: EntityId, cx: &mut Context<Self>) {
        let removed = self
            .breakpoints
            .iter()
            .filter(|(_, breakpoints_in_file)| breakpoints_in_file.buffer.entity_id() == buffer_id)
            .map(|(path, _)| path.clone())
            .collect::<Vec<_>>();
        if removed.is_empty() {
            return;
        }
        for path in &removed {
            self.breakpoints.remove(path);
            self.confirmed_remote_breakpoints.remove(path);
            self.send_breakpoints_downstream(path);
        }
        cx.emit(BreakpointStoreEvent::BreakpointsCleared(removed));
        cx.notify();
    }

    fn send_breakpoints_downstream(&self, abs_path: &Arc<Path>) {
        let Some((client, project_id)) = &self.downstream_client else {
            return;
        };
        let breakpoints = self
            .breakpoints
            .get(abs_path)
            .map(|breakpoint_set| {
                breakpoint_set
                    .breakpoints
                    .iter()
                    .map(|bp| bp.bp.to_proto(&bp.session_state))
                    .collect()
            })
            .unwrap_or_default();

        let version = self.downstream_version.get() + 1;
        self.downstream_version.set(version);
        self.downstream_path_versions
            .borrow_mut()
            .insert(abs_path.clone(), version);
        client
            .send(proto::BreakpointsForFile {
                project_id: *project_id,
                path: abs_path.to_string_lossy().into_owned(),
                breakpoints,
                version,
            })
            .log_err();
    }

    pub fn on_file_rename(
        &mut self,
        old_path: Arc<Path>,
        new_path: Arc<Path>,
        cx: &mut Context<Self>,
    ) {
        if old_path == new_path {
            return;
        }
        if let Some(confirmed) = self.confirmed_remote_breakpoints.remove(&old_path) {
            self.confirmed_remote_breakpoints
                .insert(new_path.clone(), confirmed);
        }
        for edit in &mut self.pending_upstream_edits {
            if edit.path == old_path {
                edit.path = new_path.clone();
            }
        }
        let Some(breakpoints) = self.breakpoints.remove(&old_path) else {
            return;
        };
        self.breakpoints.insert(new_path.clone(), breakpoints);
        self.send_breakpoints_downstream(&old_path);
        self.send_breakpoints_downstream(&new_path);
        cx.emit(BreakpointStoreEvent::BreakpointsCleared(vec![old_path]));
        cx.emit(BreakpointStoreEvent::BreakpointsUpdated(
            new_path,
            BreakpointUpdatedReason::Toggled,
        ));
        cx.notify();
    }

    pub fn clear_breakpoints(&mut self, cx: &mut Context<Self>) {
        let cleared = std::mem::take(&mut self.breakpoints);
        for pending in self.pending_restored.values_mut() {
            pending.clear();
        }
        self.pending_restored_buffer_changes.clear();
        for (path, breakpoints_in_file) in &cleared {
            let previous_breakpoints = breakpoint_positions(&breakpoints_in_file.breakpoints);
            self.sync_file_breakpoints(
                path,
                &breakpoints_in_file.buffer,
                &previous_breakpoints,
                cx,
            );
        }
        cx.emit(BreakpointStoreEvent::BreakpointsCleared(
            cleared.into_keys().collect(),
        ));
        cx.notify();
    }

    pub fn breakpoints<'a>(
        &'a self,
        buffer: &'a Entity<Buffer>,
        range: Option<Range<text::Anchor>>,
        buffer_snapshot: &'a BufferSnapshot,
        cx: &App,
    ) -> impl Iterator<Item = (&'a BreakpointWithPosition, Option<BreakpointSessionState>)> + 'a
    {
        let abs_path = Self::abs_path_from_buffer(buffer, cx);
        let active_session_id = self
            .active_stack_frame
            .as_ref()
            .map(|frame| frame.session_id);
        abs_path
            .and_then(|path| self.breakpoints.get(&path))
            .into_iter()
            .flat_map(move |file_breakpoints| {
                file_breakpoints.breakpoints.iter().filter_map({
                    let range = range.clone();
                    move |bp| {
                        if !buffer_snapshot.can_resolve(bp.position()) {
                            return None;
                        }

                        if let Some(range) = &range
                            && (bp.position().cmp(&range.start, buffer_snapshot).is_lt()
                                || bp.position().cmp(&range.end, buffer_snapshot).is_gt())
                        {
                            return None;
                        }
                        let session_state = active_session_id
                            .and_then(|id| bp.session_state.get(&id))
                            .copied();
                        Some((&bp.bp, session_state))
                    }
                })
            })
    }

    pub fn active_position(&self) -> Option<&ActiveStackFrame> {
        self.active_stack_frame.as_ref()
    }

    pub fn active_debug_line_pane_id(&self) -> Option<EntityId> {
        self.active_debug_line_pane_id
    }

    pub fn set_active_debug_pane_id(&mut self, pane_id: EntityId) {
        self.active_debug_line_pane_id = Some(pane_id);
    }

    pub fn remove_active_position(
        &mut self,
        session_id: Option<SessionId>,
        cx: &mut Context<Self>,
    ) {
        if let Some(session_id) = session_id {
            if self
                .active_stack_frame
                .take_if(|active_stack_frame| active_stack_frame.session_id == session_id)
                .is_some()
            {
                self.active_debug_line_pane_id = None;
            }
        } else {
            self.active_stack_frame.take();
            self.active_debug_line_pane_id = None;
        }

        cx.emit(BreakpointStoreEvent::ClearDebugLines);
        cx.notify();
    }

    pub fn set_active_position(&mut self, position: ActiveStackFrame, cx: &mut Context<Self>) {
        if self
            .active_stack_frame
            .as_ref()
            .is_some_and(|active_position| active_position == &position)
        {
            cx.emit(BreakpointStoreEvent::SetDebugLine);
            return;
        }

        if self.active_stack_frame.is_some() {
            cx.emit(BreakpointStoreEvent::ClearDebugLines);
        }

        self.active_stack_frame = Some(position);

        cx.emit(BreakpointStoreEvent::SetDebugLine);
        cx.notify();
    }

    pub fn breakpoint_with_id(
        &self,
        path: &Path,
        id: BreakpointId,
    ) -> Option<(Entity<Buffer>, BreakpointWithPosition)> {
        let breakpoints = self.breakpoints.get(path)?;
        breakpoints
            .breakpoints
            .iter()
            .find(|breakpoint| breakpoint.bp.id == id)
            .map(|breakpoint| (breakpoints.buffer.clone(), breakpoint.bp.clone()))
    }

    pub fn breakpoints_from_path(&self, path: &Arc<Path>) -> Vec<BreakpointWithPosition> {
        self.breakpoints
            .get(path)
            .map(|bp| bp.breakpoints.iter().map(|bp| bp.bp.clone()).collect())
            .unwrap_or_default()
    }

    pub fn source_breakpoints_from_path(
        &self,
        path: &Arc<Path>,
        cx: &App,
    ) -> Vec<SourceBreakpoint> {
        self.breakpoints
            .get(path)
            .map(|bp| {
                let snapshot = bp.buffer.read(cx).snapshot();
                bp.breakpoints
                    .iter()
                    .map(|bp| {
                        let position = snapshot.summary_for_anchor::<PointUtf16>(bp.position()).row;
                        let bp = &bp.bp;
                        SourceBreakpoint {
                            row: position,
                            path: path.clone(),
                            state: bp.bp.state,
                            message: bp.bp.message.clone(),
                            condition: bp.bp.condition.clone(),
                            hit_condition: bp.bp.hit_condition.clone(),
                        }
                    })
                    .collect()
            })
            .unwrap_or_default()
    }

    pub fn all_breakpoints(&self) -> BTreeMap<Arc<Path>, Vec<BreakpointWithPosition>> {
        self.breakpoints
            .iter()
            .map(|(path, bp)| {
                (
                    path.clone(),
                    bp.breakpoints.iter().map(|bp| bp.bp.clone()).collect(),
                )
            })
            .collect()
    }
    pub fn all_source_breakpoints(&self, cx: &App) -> BTreeMap<Arc<Path>, Vec<SourceBreakpoint>> {
        self.all_source_breakpoints_with_ids(cx)
            .into_iter()
            .map(|(path, breakpoints)| {
                (
                    path,
                    breakpoints
                        .into_iter()
                        .map(|(_, breakpoint)| breakpoint)
                        .collect(),
                )
            })
            .collect()
    }

    pub fn all_source_breakpoints_with_ids(
        &self,
        cx: &App,
    ) -> BTreeMap<Arc<Path>, Vec<(BreakpointId, SourceBreakpoint)>> {
        self.breakpoints
            .iter()
            .map(|(path, bp)| {
                let snapshot = bp.buffer.read(cx).snapshot();
                (
                    path.clone(),
                    bp.breakpoints
                        .iter()
                        .map(|breakpoint| {
                            let position = snapshot
                                .summary_for_anchor::<PointUtf16>(breakpoint.position())
                                .row;
                            let breakpoint = &breakpoint.bp;
                            (
                                breakpoint.id,
                                SourceBreakpoint {
                                    row: position,
                                    path: path.clone(),
                                    message: breakpoint.bp.message.clone(),
                                    state: breakpoint.bp.state,
                                    hit_condition: breakpoint.bp.hit_condition.clone(),
                                    condition: breakpoint.bp.condition.clone(),
                                },
                            )
                        })
                        .collect(),
                )
            })
            .collect()
    }

    pub fn begin_restoring(&mut self, breakpoints: BTreeMap<Arc<Path>, Vec<SourceBreakpoint>>) {
        if !self.restores_saved_breakpoints() {
            return;
        }
        for (path, restored) in breakpoints {
            if restored.is_empty() || self.restoring_paths.contains(&path) {
                continue;
            }
            self.restoring_paths.insert(path.clone());
            let pending = self.pending_restored.entry(path).or_default();
            for breakpoint in restored {
                if !pending.contains(&breakpoint) {
                    pending.push(breakpoint);
                }
            }
        }
    }

    pub fn source_breakpoints_to_persist(
        &self,
        cx: &App,
    ) -> BTreeMap<Arc<Path>, Vec<SourceBreakpoint>> {
        let mut breakpoints = self.all_source_breakpoints(cx);
        for (path, pending) in &self.pending_restored {
            let live_rows = breakpoints
                .get(path)
                .map(|live| {
                    live.iter()
                        .map(|breakpoint| breakpoint.row)
                        .collect::<collections::HashSet<_>>()
                })
                .unwrap_or_default();
            let restored = pending
                .iter()
                .filter(|breakpoint| !live_rows.contains(&breakpoint.row))
                .cloned()
                .collect::<Vec<_>>();
            if !restored.is_empty() {
                breakpoints
                    .entry(path.clone())
                    .or_default()
                    .extend(restored);
            }
        }
        breakpoints
    }

    pub fn with_serialized_breakpoints(
        &mut self,
        breakpoints: BTreeMap<Arc<Path>, Vec<SourceBreakpoint>>,
        cx: &mut Context<BreakpointStore>,
    ) -> Task<Result<()>> {
        self.begin_restoring(breakpoints);
        self.finish_restoring(cx)
    }

    pub fn restore_pending_for_paths(
        &mut self,
        project_paths: impl IntoIterator<Item = ProjectPath>,
        cx: &mut Context<Self>,
    ) {
        if self.pending_restored.is_empty() {
            return;
        }
        let buffers = project_paths
            .into_iter()
            .filter_map(|project_path| self.buffer_store.read(cx).get_by_path(&project_path))
            .collect::<Vec<_>>();
        for buffer in buffers {
            self.restore_pending_in_buffer(&buffer, cx);
        }
    }

    pub fn finish_restoring(&mut self, cx: &mut Context<BreakpointStore>) -> Task<Result<()>> {
        if !self.restores_saved_breakpoints() {
            return Task::ready(Ok(()));
        }
        let open_buffers = self
            .pending_restored
            .keys()
            .filter_map(|path| {
                let project_path = self
                    .worktree_store
                    .read(cx)
                    .project_path_for_absolute_path(path, cx)?;
                self.buffer_store.read(cx).get_by_path(&project_path)
            })
            .collect::<Vec<_>>();
        for buffer in open_buffers {
            self.restore_pending_in_buffer(&buffer, cx);
        }
        if self.pending_restored.is_empty() {
            return Task::ready(Ok(()));
        }
        self.pending_restored_buffers = Some(cx.subscribe(
            &self.buffer_store,
            |breakpoint_store, _, event, cx| {
                if let BufferStoreEvent::BufferAdded(buffer) = event {
                    breakpoint_store.restore_pending_in_buffer(buffer, cx);
                }
            },
        ));
        let paths = self.pending_restored.keys().cloned().collect::<Vec<_>>();
        let worktree_store = self.worktree_store.downgrade();
        let buffer_store = self.buffer_store.downgrade();
        cx.spawn(async move |this, cx| {
            for path in paths {
                if !this.read_with(cx, |this, _| this.pending_restored.contains_key(&path))? {
                    continue;
                }
                let buffer = async {
                    let (worktree, relative_path) = worktree_store
                        .update(cx, |this, cx| {
                            this.find_or_create_worktree(&path, false, cx)
                        })?
                        .await?;
                    buffer_store
                        .update(cx, |this, cx| {
                            let path = ProjectPath {
                                worktree_id: worktree.read(cx).id(),
                                path: relative_path,
                            };
                            this.open_buffer(path, cx)
                        })?
                        .await
                }
                .await;
                this.update(cx, |this, cx| match buffer {
                    Ok(buffer) => this.restore_pending_in_buffer(&buffer, cx),
                    Err(error) => {
                        if let Some(pending) = this.pending_restored.get(&path) {
                            log::warn!(
                                "Keeping {} breakpoints of {path:?} until it can be opened: {error:#}",
                                pending.len()
                            );
                        }
                    }
                })?;
            }
            this.update(cx, |this, _| {
                if this.pending_restored.is_empty() {
                    this.pending_restored_buffers = None;
                }
            })
        })
    }

    fn restore_pending_in_buffer(&mut self, buffer: &Entity<Buffer>, cx: &mut Context<Self>) {
        let Some(path) = Self::abs_path_from_buffer(buffer, cx) else {
            return;
        };
        if !self.pending_restored.contains_key(&path) {
            return;
        }
        let buffer_ref = buffer.read(cx);
        if buffer_ref.is_empty()
            && buffer_ref
                .file()
                .is_some_and(|file| !file.disk_state().exists())
        {
            self.pending_restored_buffer_changes.insert(
                path,
                cx.subscribe(buffer, |breakpoint_store, buffer, event, cx| {
                    if matches!(event, BufferEvent::Reloaded | BufferEvent::Saved) {
                        breakpoint_store.restore_pending_in_buffer(&buffer, cx);
                    }
                }),
            );
            return;
        }
        let Some(pending) = self.pending_restored.remove(&path) else {
            return;
        };
        self.pending_restored_buffer_changes.remove(&path);
        self.restoring_paths.remove(&path);
        let snapshot = buffer.read(cx).snapshot();
        let max_point = snapshot.max_point_utf16();
        let mut restored = BreakpointsInFile::new(buffer.clone(), cx);
        for breakpoint in pending {
            let point = PointUtf16::new(breakpoint.row, 0);
            if point > max_point {
                log::error!("skipping a deserialized breakpoint that's out of range");
                continue;
            }
            restored
                .breakpoints
                .push(StatefulBreakpoint::new(BreakpointWithPosition::new(
                    snapshot.anchor_after(point),
                    Breakpoint {
                        message: breakpoint.message,
                        state: breakpoint.state,
                        condition: breakpoint.condition,
                        hit_condition: breakpoint.hit_condition,
                    },
                )));
        }
        let previous_breakpoints = self
            .breakpoints
            .get(&path)
            .map(|live| breakpoint_positions(&live.breakpoints))
            .unwrap_or_default();
        let added = match self.breakpoints.get_mut(&path) {
            Some(live) => live.merge_restored(restored, cx),
            None => {
                let added = restored
                    .breakpoints
                    .iter()
                    .map(|breakpoint| *breakpoint.position())
                    .collect::<Vec<_>>();
                if !added.is_empty() {
                    self.breakpoints.insert(path.clone(), restored);
                }
                added
            }
        };
        if self.pending_restored.is_empty() {
            self.pending_restored_buffers = None;
        }
        if !added.is_empty() {
            let synced_buffer = self
                .breakpoints
                .get(&path)
                .map_or_else(|| buffer.clone(), |live| live.buffer.clone());
            self.sync_file_breakpoints(&path, &synced_buffer, &previous_breakpoints, cx);
            cx.emit(BreakpointStoreEvent::BreakpointsUpdated(
                path,
                BreakpointUpdatedReason::Toggled,
            ));
        }
        cx.notify();
    }

    fn restores_saved_breakpoints(&self) -> bool {
        match &self.mode {
            BreakpointStoreMode::Local => true,
            BreakpointStoreMode::Remote(remote) => !remote.upstream_client.is_via_collab(),
        }
    }

    fn forget_pending_restored(&mut self, path: &Path, row: u32) {
        if let Some(pending) = self.pending_restored.get_mut(path) {
            pending.retain(|breakpoint| breakpoint.row != row);
        }
    }

    #[cfg(any(test, feature = "test-support"))]
    pub(crate) fn breakpoint_paths(&self) -> Vec<Arc<Path>> {
        self.breakpoints.keys().cloned().collect()
    }

    #[cfg(any(test, feature = "test-support"))]
    pub fn pending_remote_edits(&self) -> (usize, usize) {
        (
            self.remote_edits.values().map(VecDeque::len).sum(),
            self.remote_edit_buffers.len(),
        )
    }
}

#[derive(Clone, Copy)]
pub enum BreakpointUpdatedReason {
    Toggled,
    FileSaved,
}

pub enum BreakpointStoreEvent {
    SetDebugLine,
    ClearDebugLines,
    BreakpointsUpdated(Arc<Path>, BreakpointUpdatedReason),
    BreakpointsCleared(Vec<Arc<Path>>),
}

impl EventEmitter<BreakpointStoreEvent> for BreakpointStore {}

type BreakpointMessage = Arc<str>;

#[derive(Clone, Debug)]
pub enum BreakpointEditAction {
    Toggle,
    InvertState,
    EditLogMessage(BreakpointMessage),
    EditCondition(BreakpointMessage),
    EditHitCondition(BreakpointMessage),
}

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub enum BreakpointState {
    Enabled,
    Disabled,
}

impl BreakpointState {
    #[inline]
    pub fn is_enabled(&self) -> bool {
        matches!(self, BreakpointState::Enabled)
    }

    #[inline]
    pub fn is_disabled(&self) -> bool {
        matches!(self, BreakpointState::Disabled)
    }

    #[inline]
    pub fn to_int(self) -> i32 {
        match self {
            BreakpointState::Enabled => 0,
            BreakpointState::Disabled => 1,
        }
    }
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct Breakpoint {
    pub message: Option<BreakpointMessage>,
    /// How many times do we hit the breakpoint until we actually stop at it e.g. (2 = 2 times of the breakpoint action)
    pub hit_condition: Option<Arc<str>>,
    pub condition: Option<BreakpointMessage>,
    pub state: BreakpointState,
}

impl Breakpoint {
    pub fn new_standard() -> Self {
        Self {
            state: BreakpointState::Enabled,
            hit_condition: None,
            condition: None,
            message: None,
        }
    }

    pub fn new_condition(hit_condition: &str) -> Self {
        Self {
            state: BreakpointState::Enabled,
            condition: None,
            hit_condition: Some(hit_condition.into()),
            message: None,
        }
    }

    pub fn new_log(log_message: &str) -> Self {
        Self {
            state: BreakpointState::Enabled,
            hit_condition: None,
            condition: None,
            message: Some(log_message.into()),
        }
    }

    fn from_proto(breakpoint: client::proto::Breakpoint) -> Option<Self> {
        Some(Self {
            state: match proto::BreakpointState::try_from(breakpoint.state).ok() {
                Some(proto::BreakpointState::Disabled) => BreakpointState::Disabled,
                None | Some(proto::BreakpointState::Enabled) => BreakpointState::Enabled,
            },
            message: breakpoint.message.map(Into::into),
            condition: breakpoint.condition.map(Into::into),
            hit_condition: breakpoint.hit_condition.map(Into::into),
        })
    }

    #[inline]
    pub fn is_enabled(&self) -> bool {
        self.state.is_enabled()
    }

    #[inline]
    pub fn is_disabled(&self) -> bool {
        self.state.is_disabled()
    }
}

impl BreakpointWithPosition {
    fn to_proto(
        &self,
        session_states: &HashMap<SessionId, BreakpointSessionState>,
    ) -> client::proto::Breakpoint {
        client::proto::Breakpoint {
            position: Some(serialize_text_anchor(&self.position)),
            state: match self.bp.state {
                BreakpointState::Enabled => proto::BreakpointState::Enabled.into(),
                BreakpointState::Disabled => proto::BreakpointState::Disabled.into(),
            },
            message: self.bp.message.as_ref().map(|s| String::from(s.as_ref())),
            condition: self.bp.condition.as_ref().map(|s| String::from(s.as_ref())),
            hit_condition: self
                .bp
                .hit_condition
                .as_ref()
                .map(|s| String::from(s.as_ref())),
            id: Some(self.id.0),
            session_state: session_states
                .iter()
                .map(|(session_id, state)| {
                    (
                        session_id.to_proto(),
                        proto::BreakpointSessionState {
                            id: state.id,
                            verified: state.verified,
                        },
                    )
                })
                .collect(),
        }
    }

    fn from_proto(mut breakpoint: proto::Breakpoint) -> Result<Self> {
        let position = language::proto::deserialize_anchor(
            breakpoint
                .position
                .take()
                .context("Anchor not present in RPC payload")?,
        )
        .context("Anchor deserialization failed")?;
        let id = breakpoint.id;
        let bp = Breakpoint::from_proto(breakpoint).context("Could not deserialize breakpoint")?;
        Ok(match id {
            Some(id) => Self {
                position,
                bp,
                id: BreakpointId(id),
            },
            None => Self::new(position, bp),
        })
    }
}

/// Breakpoint for location within source code.
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct SourceBreakpoint {
    pub row: u32,
    pub path: Arc<Path>,
    pub message: Option<Arc<str>>,
    pub condition: Option<Arc<str>>,
    pub hit_condition: Option<Arc<str>>,
    pub state: BreakpointState,
}

impl From<SourceBreakpoint> for dap::SourceBreakpoint {
    fn from(bp: SourceBreakpoint) -> Self {
        Self {
            line: bp.row as u64 + 1,
            column: None,
            condition: bp
                .condition
                .map(|condition| String::from(condition.as_ref())),
            hit_condition: bp
                .hit_condition
                .map(|hit_condition| String::from(hit_condition.as_ref())),
            log_message: bp.message.map(|message| String::from(message.as_ref())),
            mode: None,
        }
    }
}

fn upstream_edits<'a>(
    previous: &'a [BreakpointWithPosition],
    current: &'a [StatefulBreakpoint],
) -> Vec<(&'a BreakpointWithPosition, proto::BreakpointEdit)> {
    let mut edits = Vec::new();
    for previous in previous {
        match current.iter().find(|current| current.bp.id == previous.id) {
            Some(current) if current.bp == *previous => {}
            Some(current) => edits.extend([
                (previous, proto::BreakpointEdit::Superseded),
                (&current.bp, proto::BreakpointEdit::Set),
            ]),
            None => edits.push((previous, proto::BreakpointEdit::Remove)),
        }
    }
    edits.extend(
        current
            .iter()
            .filter(|current| previous.iter().all(|previous| previous.id != current.bp.id))
            .map(|current| (&current.bp, proto::BreakpointEdit::Set)),
    );
    edits
}

fn apply_upstream_edit_to(
    breakpoints: &mut Vec<StatefulBreakpoint>,
    breakpoint: BreakpointWithPosition,
    edit: proto::BreakpointEdit,
) {
    let existing = breakpoints
        .iter()
        .position(|existing| existing.bp.id == breakpoint.id);
    match (edit, existing) {
        (proto::BreakpointEdit::Set, Some(index)) => breakpoints[index].bp = breakpoint,
        (proto::BreakpointEdit::Set, None) => breakpoints.push(StatefulBreakpoint::new(breakpoint)),
        (proto::BreakpointEdit::Remove, Some(index)) => {
            breakpoints.remove(index);
        }
        _ => {}
    }
}

fn reanchor_breakpoint(
    mut breakpoint: BreakpointWithPosition,
    source: &BufferSnapshot,
    destination: &BufferSnapshot,
) -> Option<BreakpointWithPosition> {
    if source.remote_id() == destination.remote_id() {
        return Some(breakpoint);
    }
    if !source.can_resolve(&breakpoint.position) {
        log::error!(
            "Dropping breakpoint {:?}: its position does not belong to its source buffer",
            breakpoint.id
        );
        return None;
    }
    let row = source
        .summary_for_anchor::<PointUtf16>(&breakpoint.position)
        .row;
    let position = destination.clip_point_utf16(Unclipped(PointUtf16::new(row, 0)), Bias::Left);
    breakpoint.position = destination.anchor_after(position);
    Some(breakpoint)
}

fn breakpoint_positions(breakpoints: &[StatefulBreakpoint]) -> Vec<BreakpointWithPosition> {
    breakpoints
        .iter()
        .map(|breakpoint| breakpoint.bp.clone())
        .collect()
}

fn breakpoint_index(
    breakpoints: &[StatefulBreakpoint],
    breakpoint: &BreakpointWithPosition,
) -> Option<usize> {
    breakpoints
        .iter()
        .position(|existing| existing.bp.id == breakpoint.id)
        .or_else(|| {
            breakpoints
                .iter()
                .position(|existing| existing.bp.same_value(breakpoint))
        })
}

fn breakpoint_to_edit(
    breakpoints: &mut [StatefulBreakpoint],
    index: Option<usize>,
) -> Option<&mut Breakpoint> {
    breakpoints
        .get_mut(index?)
        .map(|existing| &mut existing.bp.bp)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Project;
    use fs::{FakeFs, Fs as _};
    use futures::future::BoxFuture;
    use gpui::{AppContext as _, TestAppContext};
    use itertools::Itertools;
    use parking_lot::Mutex;
    use rpc::{ProtoClient, ProtoMessageHandlerSet, proto::EnvelopedMessage as _};
    use serde_json::json;
    use settings::SettingsStore;
    use text::Point;
    use util::{path, rel_path::rel_path};

    #[gpui::test]
    async fn test_toggle_breakpoint_requests_are_gated_by_collaborators(cx: &mut TestAppContext) {
        cx.update(|cx| {
            let settings_store = SettingsStore::test(cx);
            cx.set_global(settings_store);
        });
        let fs = FakeFs::new(cx.executor());
        fs.insert_tree(path!("/a"), json!({ "test.txt": "one\ntwo" }))
            .await;
        let project = Project::test(fs, [Path::new(path!("/a"))], cx).await;
        let worktree_id = project.read_with(cx, |project, cx| {
            project.worktrees(cx).next().unwrap().read(cx).id()
        });
        let buffer = project
            .update(cx, |project, cx| {
                project.open_buffer((worktree_id, rel_path("test.txt")), cx)
            })
            .await
            .unwrap();
        let breakpoint_store = project.read_with(cx, |project, _| project.breakpoint_store());
        let guest = proto::PeerId { owner_id: 1, id: 7 };
        let not_a_collaborator = format!("Peer {guest} is not a collaborator of this project");
        breakpoint_store.update(cx, |breakpoint_store, _| {
            breakpoint_store
                .downstream_path_versions
                .borrow_mut()
                .insert(Arc::from(Path::new(path!("/a/test.txt"))), 23);
        });

        let before_joining = toggle_breakpoint_from(&breakpoint_store, &buffer, guest, cx).await;
        breakpoint_store.update(cx, |breakpoint_store, _| {
            breakpoint_store.collaborator_added(guest)
        });
        let after_joining = toggle_breakpoint_from(&breakpoint_store, &buffer, guest, cx).await;
        breakpoint_store.update(cx, |breakpoint_store, _| {
            breakpoint_store.collaborator_removed(guest)
        });
        let after_leaving = toggle_breakpoint_from(&breakpoint_store, &buffer, guest, cx).await;

        assert_eq!(
            (
                before_joining,
                after_joining,
                after_leaving,
                breakpoint_store.read_with(cx, |breakpoint_store, cx| {
                    breakpoint_store
                        .all_source_breakpoints(cx)
                        .into_values()
                        .flatten()
                        .map(|breakpoint| breakpoint.row)
                        .collect::<Vec<_>>()
                }),
            ),
            (
                Err(not_a_collaborator.clone()),
                Ok(proto::ToggleBreakpointResponse {
                    version: 0,
                    processed_breakpoints: 0
                }),
                Err(not_a_collaborator),
                Vec::from([1]),
            )
        );
    }

    #[gpui::test]
    async fn test_upstream_edits_converge_on_hosts_that_only_toggle(cx: &mut TestAppContext) {
        cx.update(|cx| {
            let settings_store = SettingsStore::test(cx);
            cx.set_global(settings_store);
        });
        let mut histories = Vec::new();
        for host_only_toggles in [true, false] {
            let fs = FakeFs::new(cx.executor());
            fs.insert_tree(path!("/a"), json!({ "test.txt": "one\ntwo" }))
                .await;
            let project = Project::test(fs, [Path::new(path!("/a"))], cx).await;
            let buffer = project
                .update(cx, |project, cx| {
                    project.open_local_buffer(path!("/a/test.txt"), cx)
                })
                .await
                .unwrap();
            let breakpoint_store = project.read_with(cx, |project, _| project.breakpoint_store());
            let enabled = BreakpointWithPosition::new(
                buffer.read_with(cx, |buffer, _| buffer.anchor_before(Point::new(1, 0))),
                Breakpoint::new_standard(),
            );
            let disabled = BreakpointWithPosition {
                bp: Breakpoint {
                    state: BreakpointState::Disabled,
                    ..enabled.bp.clone()
                },
                ..enabled.clone()
            };
            let states = [
                Vec::new(),
                Vec::from([enabled]),
                Vec::from([disabled.clone()]),
                Vec::from([disabled]),
                Vec::new(),
            ];
            let mut history = Vec::new();
            for (previous, current) in states.iter().tuple_windows() {
                let current = current
                    .iter()
                    .cloned()
                    .map(StatefulBreakpoint::new)
                    .collect::<Vec<_>>();
                for (breakpoint, edit) in upstream_edits(previous, &current) {
                    if host_only_toggles {
                        breakpoint_store.update(cx, |breakpoint_store, cx| {
                            breakpoint_store.toggle_breakpoint(
                                buffer.clone(),
                                breakpoint.clone(),
                                BreakpointEditAction::Toggle,
                                cx,
                            )
                        });
                    } else {
                        send_upstream_edit(&breakpoint_store, breakpoint, edit, cx)
                            .await
                            .unwrap();
                    }
                }
                history.push(breakpoint_store.read_with(cx, |breakpoint_store, cx| {
                    breakpoint_store
                        .all_source_breakpoints(cx)
                        .into_values()
                        .flatten()
                        .map(|breakpoint| (breakpoint.row, breakpoint.state))
                        .collect::<Vec<_>>()
                }));
            }
            histories.push(history);
        }

        let expected = Vec::from([
            Vec::from([(1, BreakpointState::Enabled)]),
            Vec::from([(1, BreakpointState::Disabled)]),
            Vec::from([(1, BreakpointState::Disabled)]),
            Vec::new(),
        ]);
        assert_eq!(histories, Vec::from([expected.clone(), expected]));
    }

    #[gpui::test]
    async fn test_upstream_edits_keep_other_breakpoints_on_a_joined_row(cx: &mut TestAppContext) {
        cx.update(|cx| {
            let settings_store = SettingsStore::test(cx);
            cx.set_global(settings_store);
        });
        let fs = FakeFs::new(cx.executor());
        fs.insert_tree(path!("/a"), json!({ "test.txt": "aaa\nbbb\n" }))
            .await;
        let project = Project::test(fs, [Path::new(path!("/a"))], cx).await;
        let buffer = project
            .update(cx, |project, cx| {
                project.open_local_buffer(path!("/a/test.txt"), cx)
            })
            .await
            .unwrap();
        let breakpoint_store = project.read_with(cx, |project, _| project.breakpoint_store());
        let [first, second] = [0, 1].map(|row| {
            BreakpointWithPosition::new(
                buffer.read_with(cx, |buffer, _| buffer.anchor_after(Point::new(row, 0))),
                Breakpoint::new_standard(),
            )
        });
        for breakpoint in [&first, &second] {
            breakpoint_store.update(cx, |breakpoint_store, cx| {
                breakpoint_store.toggle_breakpoint(
                    buffer.clone(),
                    breakpoint.clone(),
                    BreakpointEditAction::Toggle,
                    cx,
                )
            });
        }
        buffer.update(cx, |buffer, cx| {
            buffer.edit([(Point::new(0, 3)..Point::new(1, 0), "")], None, cx)
        });

        let disabled_first = BreakpointWithPosition {
            bp: Breakpoint {
                state: BreakpointState::Disabled,
                ..first.bp.clone()
            },
            ..first.clone()
        };
        let previous = [second.clone(), first];
        let current = [second, disabled_first].map(StatefulBreakpoint::new);
        let edits = upstream_edits(&previous, &current)
            .into_iter()
            .map(|(breakpoint, edit)| (breakpoint.clone(), edit))
            .collect::<Vec<_>>();
        for (breakpoint, edit) in &edits {
            send_upstream_edit(&breakpoint_store, breakpoint, *edit, cx)
                .await
                .unwrap();
        }
        buffer.update(cx, |buffer, cx| buffer.undo(cx));

        assert_eq!(
            (
                edits.into_iter().map(|(_, edit)| edit).collect::<Vec<_>>(),
                breakpoint_store.read_with(cx, |breakpoint_store, cx| {
                    breakpoint_store
                        .all_source_breakpoints(cx)
                        .into_values()
                        .flatten()
                        .map(|breakpoint| (breakpoint.row, breakpoint.state))
                        .sorted_by_key(|(row, _)| *row)
                        .collect::<Vec<_>>()
                }),
            ),
            (
                Vec::from([
                    proto::BreakpointEdit::Superseded,
                    proto::BreakpointEdit::Set
                ]),
                Vec::from([
                    (0, BreakpointState::Disabled),
                    (1, BreakpointState::Enabled)
                ]),
            )
        );
    }

    #[gpui::test]
    async fn test_snapshots_without_ids_keep_known_breakpoint_identities(cx: &mut TestAppContext) {
        cx.update(|cx| {
            let settings_store = SettingsStore::test(cx);
            cx.set_global(settings_store);
        });
        let fs = FakeFs::new(cx.executor());
        fs.insert_tree(path!("/a"), json!({ "test.txt": "one\ntwo\nthree" }))
            .await;
        let project = Project::test(fs, [Path::new(path!("/a"))], cx).await;
        let buffer = project
            .update(cx, |project, cx| {
                project.open_local_buffer(path!("/a/test.txt"), cx)
            })
            .await
            .unwrap();
        let breakpoint_store = project.read_with(cx, |project, _| project.breakpoint_store());
        let path = Arc::<Path>::from(Path::new(path!("/a/test.txt")));
        let [confirmed, pending] = [2, 1].map(|row| {
            BreakpointWithPosition::new(
                buffer.read_with(cx, |buffer, _| buffer.anchor_after(Point::new(row, 0))),
                Breakpoint::new_standard(),
            )
        });
        let host_snapshot = [
            BreakpointWithPosition {
                bp: Breakpoint {
                    state: BreakpointState::Disabled,
                    ..confirmed.bp.clone()
                },
                ..BreakpointWithPosition::new(confirmed.position, Breakpoint::new_standard())
            },
            BreakpointWithPosition::new(pending.position, pending.bp.clone()),
        ];

        let displayed = breakpoint_store.update(cx, |breakpoint_store, cx| {
            for breakpoint in [&confirmed, &pending] {
                breakpoint_store.toggle_breakpoint(
                    buffer.clone(),
                    breakpoint.clone(),
                    BreakpointEditAction::Toggle,
                    cx,
                );
            }
            breakpoint_store.push_upstream_edit(
                path.clone(),
                buffer.clone(),
                pending.clone(),
                proto::BreakpointEdit::Set,
            );
            let snapshot = buffer.read(cx).snapshot();
            let incoming = host_snapshot
                .iter()
                .cloned()
                .map(|breakpoint| (StatefulBreakpoint::new(breakpoint), false))
                .collect();
            let confirmed_breakpoints =
                breakpoint_store.adopt_known_identities(&path, &snapshot, incoming);
            breakpoint_store.apply_remote_snapshot(
                &path,
                confirmed_breakpoints,
                Some(buffer.clone()),
                1,
                cx,
            );
            breakpoint_store.breakpoints[&path]
                .breakpoints
                .iter()
                .map(|breakpoint| {
                    let row = snapshot
                        .summary_for_anchor::<PointUtf16>(breakpoint.position())
                        .row;
                    (breakpoint.bp.id, row, breakpoint.bp.bp.state)
                })
                .sorted_by_key(|(_, row, _)| *row)
                .collect::<Vec<_>>()
        });

        assert_eq!(
            displayed,
            vec![
                (pending.id, 1, BreakpointState::Enabled),
                (confirmed.id, 2, BreakpointState::Disabled),
            ]
        );
    }

    #[gpui::test]
    async fn test_upstream_edits_settle_against_host_snapshot_versions(cx: &mut TestAppContext) {
        cx.update(|cx| {
            let settings_store = SettingsStore::test(cx);
            cx.set_global(settings_store);
        });
        let fs = FakeFs::new(cx.executor());
        fs.insert_tree(path!("/a"), json!({ "test.txt": "one\ntwo\nthree" }))
            .await;
        let project = Project::test(fs, [Path::new(path!("/a"))], cx).await;
        let buffer = project
            .update(cx, |project, cx| {
                project.open_local_buffer(path!("/a/test.txt"), cx)
            })
            .await
            .unwrap();
        let breakpoint_store = project.read_with(cx, |project, _| project.breakpoint_store());
        let path = Arc::<Path>::from(Path::new(path!("/a/test.txt")));
        let [kept, added, rejected] = [0, 1, 2].map(|row| {
            BreakpointWithPosition::new(
                buffer.read_with(cx, |buffer, _| buffer.anchor_after(Point::new(row, 0))),
                Breakpoint::new_standard(),
            )
        });

        let displayed_rows = breakpoint_store.update(cx, |breakpoint_store, cx| {
            let mut displayed_rows = Vec::new();
            let mut record = |breakpoint_store: &BreakpointStore, cx: &App| {
                let snapshot = buffer.read(cx).snapshot();
                displayed_rows.push(
                    breakpoint_store
                        .breakpoints
                        .get(&path)
                        .into_iter()
                        .flat_map(|file_breakpoints| &file_breakpoints.breakpoints)
                        .map(|breakpoint| {
                            snapshot
                                .summary_for_anchor::<PointUtf16>(breakpoint.position())
                                .row
                        })
                        .sorted()
                        .collect::<Vec<_>>(),
                );
            };
            let snapshot_of = |breakpoints: &[&BreakpointWithPosition]| {
                breakpoints
                    .iter()
                    .map(|breakpoint| StatefulBreakpoint::new((*breakpoint).clone()))
                    .collect::<Vec<_>>()
            };
            let edit_locally = |breakpoint_store: &mut BreakpointStore,
                                breakpoint: &BreakpointWithPosition,
                                edit: proto::BreakpointEdit,
                                cx: &mut Context<BreakpointStore>| {
                breakpoint_store.toggle_breakpoint(
                    buffer.clone(),
                    breakpoint.clone(),
                    BreakpointEditAction::Toggle,
                    cx,
                );
                breakpoint_store.push_upstream_edit(
                    path.clone(),
                    buffer.clone(),
                    breakpoint.clone(),
                    edit,
                )
            };
            let apply_snapshot =
                |breakpoint_store: &mut BreakpointStore,
                 breakpoints: &[&BreakpointWithPosition],
                 version: u64,
                 cx: &mut Context<BreakpointStore>| {
                    breakpoint_store.apply_remote_snapshot(
                        &path,
                        snapshot_of(breakpoints),
                        Some(buffer.clone()),
                        version,
                        cx,
                    );
                };

            apply_snapshot(breakpoint_store, &[&kept], 1, cx);
            let addition = edit_locally(breakpoint_store, &added, proto::BreakpointEdit::Set, cx);
            apply_snapshot(breakpoint_store, &[&kept], 2, cx);
            breakpoint_store.finish_upstream_edits(&[addition], Some(3), cx);
            record(breakpoint_store, cx);
            apply_snapshot(breakpoint_store, &[&kept, &added], 3, cx);
            record(breakpoint_store, cx);

            let removal = edit_locally(breakpoint_store, &added, proto::BreakpointEdit::Remove, cx);
            record(breakpoint_store, cx);
            breakpoint_store.finish_upstream_edits(&[removal], None, cx);
            record(breakpoint_store, cx);

            let late_ack =
                edit_locally(breakpoint_store, &rejected, proto::BreakpointEdit::Set, cx);
            apply_snapshot(breakpoint_store, &[&kept, &added, &rejected], 5, cx);
            breakpoint_store.finish_upstream_edits(&[late_ack], Some(4), cx);
            record(breakpoint_store, cx);

            let rejection = edit_locally(
                breakpoint_store,
                &rejected,
                proto::BreakpointEdit::Remove,
                cx,
            );
            breakpoint_store.finish_upstream_edits(&[rejection], None, cx);
            record(breakpoint_store, cx);

            apply_snapshot(breakpoint_store, &[], 0, cx);
            let legacy_addition =
                edit_locally(breakpoint_store, &added, proto::BreakpointEdit::Set, cx);
            breakpoint_store.finish_upstream_edits(&[legacy_addition], Some(0), cx);
            record(breakpoint_store, cx);
            apply_snapshot(breakpoint_store, &[&added], 0, cx);
            record(breakpoint_store, cx);
            let legacy_removal =
                edit_locally(breakpoint_store, &added, proto::BreakpointEdit::Remove, cx);
            apply_snapshot(breakpoint_store, &[], 0, cx);
            breakpoint_store.finish_upstream_edits(&[legacy_removal], Some(0), cx);
            record(breakpoint_store, cx);
            apply_snapshot(breakpoint_store, &[&kept], 0, cx);
            record(breakpoint_store, cx);
            let conflicting_legacy_addition =
                edit_locally(breakpoint_store, &added, proto::BreakpointEdit::Set, cx);
            apply_snapshot(breakpoint_store, &[&kept], 0, cx);
            breakpoint_store.finish_upstream_edits(&[conflicting_legacy_addition], Some(0), cx);
            record(breakpoint_store, cx);
            (
                displayed_rows,
                breakpoint_store.pending_upstream_edits.len(),
            )
        });

        assert_eq!(
            displayed_rows,
            (
                vec![
                    vec![0, 1],
                    vec![0, 1],
                    vec![0],
                    vec![0, 1],
                    vec![0, 1, 2],
                    vec![0, 1, 2],
                    vec![1],
                    vec![1],
                    Vec::new(),
                    vec![0],
                    vec![0],
                ],
                0
            )
        );
    }

    #[gpui::test]
    async fn test_upstream_edits_follow_their_buffer_across_replacement_and_rename(
        cx: &mut TestAppContext,
    ) {
        cx.update(|cx| {
            let settings_store = SettingsStore::test(cx);
            cx.set_global(settings_store);
        });
        let fs = FakeFs::new(cx.executor());
        fs.insert_tree(path!("/a"), json!({ "test.txt": "one\ntwo\nthree" }))
            .await;
        let project = Project::test(fs.clone(), [Path::new(path!("/a"))], cx).await;
        let buffer = project
            .update(cx, |project, cx| {
                project.open_local_buffer(path!("/a/test.txt"), cx)
            })
            .await
            .unwrap();
        let replacement = project.update(cx, |project, cx| {
            project.create_local_buffer("zero\none\ntwo\nthree", None, false, cx)
        });
        let breakpoint_store = project.read_with(cx, |project, _| project.breakpoint_store());
        let path = Arc::<Path>::from(Path::new(path!("/a/test.txt")));
        let [confirmed, pending] = [1, 2].map(|row| {
            BreakpointWithPosition::new(
                buffer.read_with(cx, |buffer, _| buffer.anchor_after(Point::new(row, 0))),
                Breakpoint::new_standard(),
            )
        });

        let (rows_in_replacement, edit_id) = breakpoint_store.update(cx, |breakpoint_store, cx| {
            breakpoint_store.apply_remote_snapshot(
                &path,
                vec![StatefulBreakpoint::new(confirmed)],
                Some(buffer.clone()),
                1,
                cx,
            );
            let edit_id = breakpoint_store.push_upstream_edit(
                path.clone(),
                buffer.clone(),
                pending,
                proto::BreakpointEdit::Set,
            );
            breakpoint_store.apply_confirmed_remote_breakpoints(
                &path,
                Some(replacement.clone()),
                cx,
            );
            let snapshot = replacement.read(cx).snapshot();
            let rows = breakpoint_store.breakpoints[&path]
                .breakpoints
                .iter()
                .map(|breakpoint| {
                    snapshot
                        .summary_for_anchor::<PointUtf16>(breakpoint.position())
                        .row
                })
                .sorted()
                .collect::<Vec<_>>();
            (rows, edit_id)
        });
        fs.rename(
            Path::new(path!("/a/test.txt")),
            Path::new(path!("/a/renamed.txt")),
            fs::RenameOptions::default(),
        )
        .await
        .unwrap();
        cx.run_until_parked();
        let request_path = breakpoint_store.update(cx, |breakpoint_store, cx| {
            breakpoint_store
                .upstream_request(edit_id, 0, cx)
                .map(|request| request.path)
        });

        assert_eq!(
            (rows_in_replacement, request_path),
            (vec![1, 2], Some(path!("/a/renamed.txt").to_string()))
        );
    }

    #[gpui::test]
    async fn test_restoring_keeps_distinct_breakpoints_saved_on_one_row(cx: &mut TestAppContext) {
        cx.update(|cx| {
            let settings_store = SettingsStore::test(cx);
            cx.set_global(settings_store);
        });
        let fs = FakeFs::new(cx.executor());
        fs.insert_tree(path!("/a"), json!({ "test.txt": "first();second();\n" }))
            .await;
        let project = Project::test(fs, [Path::new(path!("/a"))], cx).await;
        project
            .update(cx, |project, cx| {
                project.open_local_buffer(path!("/a/test.txt"), cx)
            })
            .await
            .unwrap();
        let breakpoint_store = project.read_with(cx, |project, _| project.breakpoint_store());
        let path = Arc::<Path>::from(Path::new(path!("/a/test.txt")));
        let saved = |condition: &str| SourceBreakpoint {
            row: 0,
            path: path.clone(),
            message: None,
            condition: Some(condition.into()),
            hit_condition: None,
            state: BreakpointState::Enabled,
        };

        breakpoint_store
            .update(cx, |breakpoint_store, cx| {
                breakpoint_store.with_serialized_breakpoints(
                    BTreeMap::from([(
                        path.clone(),
                        Vec::from([saved("flag"), saved("!flag"), saved("flag")]),
                    )]),
                    cx,
                )
            })
            .await
            .unwrap();

        assert_eq!(
            breakpoint_store.read_with(cx, |breakpoint_store, cx| {
                breakpoint_store
                    .all_source_breakpoints(cx)
                    .into_values()
                    .flatten()
                    .map(|breakpoint| (breakpoint.row, breakpoint.condition))
                    .sorted()
                    .collect::<Vec<_>>()
            }),
            Vec::from([(0, Some("!flag".into())), (0, Some("flag".into()))])
        );
    }

    #[gpui::test]
    async fn test_restoring_missing_file_keeps_breakpoints_until_reload(cx: &mut TestAppContext) {
        cx.update(|cx| {
            let settings_store = SettingsStore::test(cx);
            cx.set_global(settings_store);
        });
        let fs = FakeFs::new(cx.executor());
        fs.insert_tree(path!("/a"), json!({})).await;
        let project = Project::test(fs.clone(), [Path::new(path!("/a"))], cx).await;
        let breakpoint_store = project.read_with(cx, |project, _| project.breakpoint_store());
        let path = Arc::<Path>::from(Path::new(path!("/a/test.txt")));
        let saved = BTreeMap::from([(
            path.clone(),
            [0, 1]
                .map(|row| SourceBreakpoint {
                    row,
                    path: path.clone(),
                    message: None,
                    condition: None,
                    hit_condition: None,
                    state: BreakpointState::Enabled,
                })
                .to_vec(),
        )]);
        let buffer = project
            .update(cx, |project, cx| {
                project.open_local_buffer(path!("/a/test.txt"), cx)
            })
            .await
            .unwrap();
        breakpoint_store
            .update(cx, |breakpoint_store, cx| {
                breakpoint_store.with_serialized_breakpoints(saved.clone(), cx)
            })
            .await
            .unwrap();
        assert_eq!(
            breakpoint_store.read_with(cx, |breakpoint_store, cx| (
                breakpoint_store.all_source_breakpoints(cx),
                breakpoint_store.source_breakpoints_to_persist(cx),
            )),
            (BTreeMap::new(), saved.clone())
        );

        fs.insert_tree(path!("/a"), json!({ "test.txt": "one\ntwo" }))
            .await;
        buffer
            .update(cx, |buffer, cx| buffer.reload(cx))
            .await
            .unwrap();
        cx.run_until_parked();
        assert_eq!(
            breakpoint_store.read_with(cx, |breakpoint_store, cx| {
                breakpoint_store.all_source_breakpoints(cx)
            }),
            saved
        );
    }

    #[gpui::test]
    async fn test_upstream_sets_keep_breakpoints_sharing_an_anchor(cx: &mut TestAppContext) {
        let (_project, buffer, breakpoint_store) = open_test_buffer(cx).await;
        let [flag, not_flag] = conditional_breakpoints_sharing_an_anchor(&buffer, cx);

        for breakpoint in [&flag, &not_flag] {
            send_upstream_edit(
                &breakpoint_store,
                breakpoint,
                proto::BreakpointEdit::Set,
                cx,
            )
            .await
            .unwrap();
        }

        assert_eq!(
            breakpoint_store.read_with(cx, |breakpoint_store, _| {
                breakpoint_store.breakpoints_from_path(&Arc::from(Path::new(path!("/a/test.txt"))))
            }),
            Vec::from([flag, not_flag])
        );
    }

    #[gpui::test]
    async fn test_upstream_removal_keeps_other_breakpoint_sharing_an_anchor(
        cx: &mut TestAppContext,
    ) {
        let (_project, buffer, breakpoint_store) = open_test_buffer(cx).await;
        let [flag, not_flag] = conditional_breakpoints_sharing_an_anchor(&buffer, cx);
        for breakpoint in [&flag, &not_flag] {
            breakpoint_store.update(cx, |breakpoint_store, cx| {
                breakpoint_store.toggle_breakpoint(
                    buffer.clone(),
                    breakpoint.clone(),
                    BreakpointEditAction::Toggle,
                    cx,
                )
            });
        }

        let previous = [flag, not_flag.clone()];
        let current = [StatefulBreakpoint::new(not_flag.clone())];
        for (breakpoint, edit) in upstream_edits(&previous, &current) {
            send_upstream_edit(&breakpoint_store, breakpoint, edit, cx)
                .await
                .unwrap();
        }

        assert_eq!(
            breakpoint_store.read_with(cx, |breakpoint_store, _| {
                breakpoint_store.breakpoints_from_path(&Arc::from(Path::new(path!("/a/test.txt"))))
            }),
            Vec::from([not_flag])
        );
    }

    #[gpui::test]
    async fn test_concurrent_upstream_edits_of_one_breakpoint_converge(cx: &mut TestAppContext) {
        let (_project, buffer, breakpoint_store) = open_test_buffer(cx).await;
        let [flag, not_flag] = conditional_breakpoints_sharing_an_anchor(&buffer, cx);
        for breakpoint in [&flag, &not_flag] {
            breakpoint_store.update(cx, |breakpoint_store, cx| {
                breakpoint_store.toggle_breakpoint(
                    buffer.clone(),
                    breakpoint.clone(),
                    BreakpointEditAction::Toggle,
                    cx,
                )
            });
        }
        let previous = [flag.clone(), not_flag.clone()];
        let edited_condition = BreakpointWithPosition {
            bp: Breakpoint {
                condition: Some(Arc::from("x")),
                ..not_flag.bp.clone()
            },
            ..not_flag.clone()
        };
        let disabled = BreakpointWithPosition {
            bp: Breakpoint {
                state: BreakpointState::Disabled,
                ..not_flag.bp.clone()
            },
            ..not_flag
        };

        for concurrent_edit in [&edited_condition, &disabled] {
            let current = [flag.clone(), concurrent_edit.clone()].map(StatefulBreakpoint::new);
            for (breakpoint, edit) in upstream_edits(&previous, &current) {
                send_upstream_edit(&breakpoint_store, breakpoint, edit, cx)
                    .await
                    .unwrap();
            }
        }

        assert_eq!(
            breakpoint_store.read_with(cx, |breakpoint_store, _| {
                breakpoint_store.breakpoints_from_path(&Arc::from(Path::new(path!("/a/test.txt"))))
            }),
            Vec::from([flag, disabled])
        );
    }

    #[gpui::test]
    async fn test_restored_breakpoint_batches_bound_requests_and_converge(cx: &mut TestAppContext) {
        cx.update(|cx| {
            let settings = SettingsStore::test(cx);
            cx.set_global(settings);
        });
        for (breakpoint_count, strip_batch, legacy_toggle, clear) in [
            (0, false, false, false),
            (1, false, false, false),
            (1000, false, false, false),
            (3, false, false, true),
            (3, true, true, false),
            (3, true, false, false),
        ] {
            for acknowledge_first in [false, true] {
                let fs = FakeFs::new(cx.executor());
                fs.insert_tree(
                    path!("/a"),
                    json!({ "test.txt": "line\n".repeat(breakpoint_count as usize + 1) }),
                )
                .await;
                let project = Project::test(fs, [Path::new(path!("/a"))], cx).await;
                let buffer = project
                    .update(cx, |project, cx| {
                        project.open_local_buffer(path!("/a/test.txt"), cx)
                    })
                    .await
                    .unwrap();
                let host = project.read_with(cx, |project, _| project.breakpoint_store());
                let (requests, mut incoming) = mpsc::unbounded();
                let transport = Arc::new(BreakpointTestClient {
                    requests,
                    snapshots: Mutex::default(),
                    handlers: Mutex::default(),
                });
                let client = host.update(cx, |host, cx| {
                    host.shared(0, AnyProtoClient::from(transport.clone()));
                    cx.new(|_| {
                        BreakpointStore::remote(
                            0,
                            AnyProtoClient::from(transport.clone()),
                            host.buffer_store.clone(),
                            host.worktree_store.clone(),
                        )
                    })
                });
                let path = Arc::<Path>::from(Path::new(path!("/a/test.txt")));
                let saved = (0..breakpoint_count)
                    .map(|row| SourceBreakpoint {
                        row,
                        path: path.clone(),
                        message: None,
                        condition: None,
                        hit_condition: None,
                        state: BreakpointState::Enabled,
                    })
                    .collect::<Vec<_>>();
                client
                    .update(cx, |client, cx| {
                        client.with_serialized_breakpoints(
                            BTreeMap::from([(path.clone(), saved.clone())]),
                            cx,
                        )
                    })
                    .await
                    .unwrap();
                let expected_requests = if breakpoint_count == 0 {
                    0
                } else if strip_batch {
                    breakpoint_count + 1
                } else {
                    2
                };
                let expected = if breakpoint_count <= 1 || clear {
                    BTreeMap::new()
                } else {
                    BTreeMap::from([(
                        path.clone(),
                        saved
                            .into_iter()
                            .filter(|breakpoint| breakpoint.row != 0)
                            .collect::<Vec<_>>(),
                    )])
                };
                let mut request_sizes = Vec::new();
                let mut snapshot_sizes = Vec::new();
                for request_index in 0..expected_requests {
                    let (request, reply) = incoming.next().await.unwrap();
                    let mut request = proto::ToggleBreakpoint::from_envelope(request).unwrap();
                    request_sizes.push(1 + request.additional_breakpoints.len());
                    if request_index == 0 {
                        client.update(cx, |client, cx| {
                            if clear {
                                client.clear_breakpoints(cx);
                            } else {
                                let position = buffer.read(cx).anchor_after(Point::new(0, 0));
                                client.toggle_breakpoint(
                                    buffer.clone(),
                                    BreakpointWithPosition::new(
                                        position,
                                        Breakpoint::new_standard(),
                                    ),
                                    BreakpointEditAction::Toggle,
                                    cx,
                                );
                            }
                        });
                    }
                    if strip_batch {
                        request.additional_breakpoints.clear();
                    }
                    if legacy_toggle {
                        request.edit = i32::from(proto::BreakpointEdit::Toggle);
                        request.breakpoint.as_mut().unwrap().id = None;
                    }
                    let mut response = BreakpointStore::handle_toggle_breakpoint(
                        host.clone(),
                        breakpoint_test_envelope(request),
                        cx.to_async(),
                    )
                    .await
                    .unwrap();
                    if strip_batch {
                        response.processed_breakpoints = 0;
                    }
                    let mut snapshots = std::mem::take(&mut *transport.snapshots.lock());
                    assert_eq!(snapshots.len(), 1);
                    let mut snapshot =
                        proto::BreakpointsForFile::from_envelope(snapshots.pop().unwrap()).unwrap();
                    snapshot_sizes.push(snapshot.breakpoints.len());
                    if legacy_toggle {
                        snapshot.version = 0;
                        for breakpoint in &mut snapshot.breakpoints {
                            breakpoint.id = None;
                        }
                    }
                    let mut reply = Some(reply);
                    let response = response.into_envelope(0, Some(0), None);
                    if acknowledge_first {
                        reply.take().unwrap().send(response.clone()).unwrap();
                        cx.run_until_parked();
                    }
                    BreakpointStore::handle_breakpoints_for_file(
                        client.clone(),
                        breakpoint_test_envelope(snapshot),
                        cx.to_async(),
                    )
                    .await
                    .unwrap();
                    if let Some(reply) = reply {
                        reply.send(response).unwrap();
                    }
                    cx.run_until_parked();
                    if !legacy_toggle {
                        assert_eq!(
                            client.read_with(cx, |client, cx| client
                                .source_breakpoints_to_persist(cx)),
                            expected
                        );
                    }
                }
                cx.run_until_parked();
                assert!(incoming.try_recv().is_err());
                assert_eq!(transport.snapshots.lock().len(), 0);
                let expected_request_sizes = if breakpoint_count == 0 {
                    Vec::new()
                } else if clear {
                    vec![breakpoint_count as usize; 2]
                } else {
                    std::iter::once(breakpoint_count as usize)
                        .chain(std::iter::repeat_n(1, expected_requests as usize - 1))
                        .collect::<Vec<_>>()
                };
                let expected_snapshot_sizes = if breakpoint_count == 0 {
                    Vec::new()
                } else if clear {
                    vec![breakpoint_count as usize, 0]
                } else if strip_batch {
                    (1..=breakpoint_count as usize)
                        .chain([breakpoint_count as usize - 1])
                        .collect::<Vec<_>>()
                } else {
                    vec![breakpoint_count as usize, breakpoint_count as usize - 1]
                };
                assert_eq!(
                    (request_sizes, snapshot_sizes),
                    (expected_request_sizes, expected_snapshot_sizes)
                );
                if !strip_batch {
                    let foreign_buffer = project.update(cx, |project, cx| {
                        project.create_local_buffer("other", None, false, cx)
                    });
                    let breakpoints =
                        [(&buffer, breakpoint_count), (&foreign_buffer, 0)].map(|(buffer, row)| {
                            BreakpointWithPosition::new(
                                buffer.read_with(cx, |buffer, _| {
                                    buffer.anchor_after(Point::new(row, 0))
                                }),
                                Breakpoint::new_standard(),
                            )
                            .to_proto(&HashMap::default())
                        });
                    let [first, invalid] = breakpoints;
                    for pending_first in [false, true] {
                        let mut first = first.clone();
                        if pending_first {
                            let anchor = first.position.as_mut().unwrap();
                            anchor.replica_id = u32::from(
                                buffer.read_with(cx, |buffer, _| buffer.replica_id().as_u16()),
                            ) + 1;
                            anchor.timestamp = u32::MAX;
                        }
                        let rejected = BreakpointStore::handle_toggle_breakpoint(
                            host.clone(),
                            breakpoint_test_envelope(proto::ToggleBreakpoint {
                                project_id: 0,
                                path: path.to_string_lossy().into_owned(),
                                breakpoint: Some(first),
                                edit: i32::from(proto::BreakpointEdit::Set),
                                additional_breakpoints: vec![invalid.clone()],
                            }),
                            cx.to_async(),
                        )
                        .await;
                        assert!(rejected.is_err());
                        assert_eq!(transport.snapshots.lock().len(), 0);
                        assert_eq!(
                            host.read_with(cx, |host, _| host.pending_remote_edits()),
                            (0, 0)
                        );
                    }
                }
                assert_eq!(
                    host.read_with(cx, |host, cx| host.all_source_breakpoints(cx)),
                    expected
                );
                assert_eq!(
                    client.read_with(cx, |client, cx| client.all_source_breakpoints(cx)),
                    expected
                );
                assert_eq!(
                    client.read_with(cx, |client, _| client.pending_upstream_edits.len()),
                    0
                );
            }
        }
    }

    async fn send_upstream_edit(
        breakpoint_store: &Entity<BreakpointStore>,
        breakpoint: &BreakpointWithPosition,
        edit: proto::BreakpointEdit,
        cx: &mut TestAppContext,
    ) -> Result<proto::ToggleBreakpointResponse> {
        let envelope = TypedEnvelope {
            sender_id: proto::PeerId { owner_id: 0, id: 0 },
            original_sender_id: None,
            message_id: 0,
            payload: proto::ToggleBreakpoint {
                project_id: 0,
                path: String::from(path!("/a/test.txt")),
                breakpoint: Some(breakpoint.to_proto(&HashMap::default())),
                edit: i32::from(edit),
                additional_breakpoints: Vec::new(),
            },
            received_at: std::time::Instant::now(),
        };
        BreakpointStore::handle_toggle_breakpoint(breakpoint_store.clone(), envelope, cx.to_async())
            .await
    }

    async fn toggle_breakpoint_from(
        breakpoint_store: &Entity<BreakpointStore>,
        buffer: &Entity<Buffer>,
        sender: proto::PeerId,
        cx: &mut TestAppContext,
    ) -> Result<proto::ToggleBreakpointResponse, String> {
        let position = buffer.read_with(cx, |buffer, _| buffer.anchor_before(Point::new(1, 0)));
        let envelope = TypedEnvelope {
            sender_id: proto::PeerId { owner_id: 0, id: 0 },
            original_sender_id: Some(sender),
            message_id: 0,
            payload: proto::ToggleBreakpoint {
                project_id: 0,
                path: String::from(path!("/a/test.txt")),
                breakpoint: Some(
                    BreakpointWithPosition::new(position, Breakpoint::new_standard())
                        .to_proto(&HashMap::default()),
                ),
                edit: i32::from(proto::BreakpointEdit::Toggle),
                additional_breakpoints: Vec::new(),
            },
            received_at: std::time::Instant::now(),
        };
        BreakpointStore::handle_toggle_breakpoint(breakpoint_store.clone(), envelope, cx.to_async())
            .await
            .map_err(|error| error.to_string())
    }

    async fn open_test_buffer(
        cx: &mut TestAppContext,
    ) -> (Entity<Project>, Entity<Buffer>, Entity<BreakpointStore>) {
        cx.update(|cx| {
            let settings_store = SettingsStore::test(cx);
            cx.set_global(settings_store);
        });
        let fs = FakeFs::new(cx.executor());
        fs.insert_tree(path!("/a"), json!({ "test.txt": "first();second();\n" }))
            .await;
        let project = Project::test(fs, [Path::new(path!("/a"))], cx).await;
        let buffer = project
            .update(cx, |project, cx| {
                project.open_local_buffer(path!("/a/test.txt"), cx)
            })
            .await
            .unwrap();
        let breakpoint_store = project.read_with(cx, |project, _| project.breakpoint_store());
        (project, buffer, breakpoint_store)
    }

    fn conditional_breakpoints_sharing_an_anchor(
        buffer: &Entity<Buffer>,
        cx: &mut TestAppContext,
    ) -> [BreakpointWithPosition; 2] {
        let position = buffer.read_with(cx, |buffer, _| buffer.anchor_after(Point::new(0, 0)));
        ["flag", "!flag"].map(|condition| {
            BreakpointWithPosition::new(
                position,
                Breakpoint {
                    condition: Some(Arc::from(condition)),
                    ..Breakpoint::new_standard()
                },
            )
        })
    }

    fn breakpoint_test_envelope<T>(payload: T) -> TypedEnvelope<T> {
        TypedEnvelope {
            sender_id: proto::PeerId { owner_id: 0, id: 0 },
            original_sender_id: None,
            message_id: 0,
            payload,
            received_at: std::time::Instant::now(),
        }
    }

    struct BreakpointTestClient {
        requests: mpsc::UnboundedSender<(proto::Envelope, oneshot::Sender<proto::Envelope>)>,
        snapshots: Mutex<Vec<proto::Envelope>>,
        handlers: Mutex<ProtoMessageHandlerSet>,
    }

    impl ProtoClient for BreakpointTestClient {
        fn request(
            &self,
            envelope: proto::Envelope,
            _: &'static str,
        ) -> BoxFuture<'static, Result<proto::Envelope>> {
            let (reply, response) = oneshot::channel();
            self.requests.unbounded_send((envelope, reply)).unwrap();
            Box::pin(async move { Ok(response.await?) })
        }

        fn send(&self, envelope: proto::Envelope, _: &'static str) -> Result<()> {
            self.snapshots.lock().push(envelope);
            Ok(())
        }

        fn send_response(
            &self,
            envelope: proto::Envelope,
            message_type: &'static str,
        ) -> Result<()> {
            self.send(envelope, message_type)
        }

        fn message_handler_set(&self) -> &Mutex<ProtoMessageHandlerSet> {
            &self.handlers
        }

        fn is_via_collab(&self) -> bool {
            false
        }

        fn has_wsl_interop(&self) -> bool {
            false
        }
    }
}
