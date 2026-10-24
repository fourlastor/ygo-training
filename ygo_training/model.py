"""Dynamic-action policy/value network over one decision of a seat."""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor, nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from .card_table import CODES, ROWS, TABLE, CardTable
from .config import BOARD_REPRESENTATIONS, LEAVE_OUT, left_out
from .schema import (
    ACTION_KIND_VOCAB,
    ACTION_SCALAR_COUNT,
    CARD_COMPONENT_VOCAB,
    CARD_STATUS_COUNT,
    CATEGORY_COUNT,
    CONTEXT_COUNT,
    DECISION_KIND_VOCAB,
    HINT_VOCAB,
    HISTORY_SCALAR_COUNT,
    LOCATION_VOCAB,
    NUMERIC_COUNT,
    PHASE_VOCAB,
    POSITION_VOCAB,
    SCALAR_COUNT,
)

# The environment's numbers for the two zones of the field.
MONSTER_ZONE = 3
SPELL_TRAP_ZONE = 4


class CardIdentityEmbedding(nn.Module):
    """Collision-free uint32 card identity using two 16-bit components."""

    def __init__(self, embedding_dim: int):
        super().__init__()
        low_dim = embedding_dim // 2
        high_dim = embedding_dim - low_dim
        self.low = nn.Embedding(CARD_COMPONENT_VOCAB, low_dim)
        self.high = nn.Embedding(CARD_COMPONENT_VOCAB, high_dim)

    def forward(self, card: Tensor) -> Tensor:
        return torch.cat(
            (
                self.low(card & 0xFFFF),
                self.high((card >> 16) & 0xFFFF),
            ),
            dim=-1,
        )


class PolicyValueNet(nn.Module):
    """Scores only the answers a decision offers.

    Its input is the batch the library builds of decisions, described in
    ``schema.py``.  Every card, wherever it appears (in the state, in the
    decision's context, as the card or a member of an answer), goes through
    one encoder of a card where it lies.

    From format 16 a model may be told more of a card than its input gives,
    by the code the input carries (``card_table``): with ``descriptions`` a
    fixed vector of what the card's text says, through a layer of its own
    and added to the card's identity wherever the identity is read; with
    ``hide_identities`` the identity of that share of the cards hidden in
    each update of training (``hide``), and at play the identity of every
    card it was not trained on (``play``), which it keeps.
    """

    def __init__(
        self,
        embedding_dim: int = 96,
        hidden_dim: int = 192,
        *,
        board_representation: str = "mean-pool",
        leave_out: tuple[str, ...] = (),
        cards: CardTable | None = None,
        descriptions: bool = False,
        hide_identities: float = 0.0,
    ):
        super().__init__()
        if board_representation not in BOARD_REPRESENTATIONS:
            raise ValueError(f"Unknown board representation {board_representation!r}")
        if any(part not in LEAVE_OUT for part in leave_out):
            raise ValueError(f"A model leaves out only {', '.join(LEAVE_OUT)}, not {leave_out!r}")
        self.board_representation = board_representation
        # What the model does not read of its input: each part is given
        # nothing where it was, and the network keeps every weight it has,
        # so that every model begins as the same network under one seed and
        # is set to read what its checkpoint was taught to (`checkpoints`).
        self.leave_out = left_out(leave_out)
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.card_identity = CardIdentityEmbedding(embedding_dim)
        self.card_kind_embedding = nn.Embedding(4, 8)
        self.attribute_embedding = nn.Embedding(16, 8)
        self.race_embedding = nn.Embedding(64, 12)
        self.level_embedding = nn.Embedding(16, 8)
        self.setcode_embedding = nn.Embedding(4096, 16)
        self.card_numeric = nn.Sequential(
            nn.Linear(NUMERIC_COUNT, 32), nn.ReLU(), nn.Linear(32, 24), nn.ReLU()
        )
        self.location_embedding = nn.Embedding(LOCATION_VOCAB, 16)
        self.position_embedding = nn.Embedding(POSITION_VOCAB, 12)
        self.controller_embedding = nn.Embedding(2, 4)
        self.phase_embedding = nn.Embedding(PHASE_VOCAB, 12)
        self.decision_kind_embedding = nn.Embedding(DECISION_KIND_VOCAB, 8)
        self.hint_embedding = nn.Embedding(HINT_VOCAB, 8)
        self.action_kind_embedding = nn.Embedding(ACTION_KIND_VOCAB, 24)
        self.card_representation_dim = embedding_dim + 8 + 8 + 12 + 8 + 16 + 24
        card = self.card_representation_dim
        self.placed_card_encoder = nn.Sequential(
            nn.Linear(card + 16 + 12 + 4 + CARD_STATUS_COUNT, card),
            nn.ReLU(),
        )
        self.context_encoder = nn.Sequential(nn.Linear(card, 48), nn.ReLU())
        self.action_member_encoder = nn.Sequential(nn.Linear(card + 3, embedding_dim), nn.ReLU())
        self.history = nn.LSTM(
            embedding_dim + 16 + 24 + HISTORY_SCALAR_COUNT, hidden_dim, batch_first=True
        )
        self.state = nn.Sequential(
            nn.Linear(
                card + 12 + 8 + 8 + SCALAR_COUNT + CONTEXT_COUNT * 48 + hidden_dim,
                hidden_dim,
            ),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.action = nn.Sequential(
            nn.Linear(card + 24 + ACTION_SCALAR_COUNT + embedding_dim + 1, hidden_dim),
            nn.ReLU(),
        )
        # Both representations own these parameters, so that their arms are
        # matched: pooling bypasses them, while tokens let each answer attend
        # to every card on the field.
        self.board_query = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.board_key = nn.Linear(card, hidden_dim, bias=False)
        self.board_value = nn.Linear(card, hidden_dim, bias=False)
        self.board_fusion = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.score = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)
        )
        self.value = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1), nn.Tanh()
        )
        # The lead an answer is predicted to have over the pilot's (see
        # `evaluate`).  The last parameters of the model: a checkpoint saved
        # before the head existed holds every one before it, in its place.
        self.lead = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)
        )
        # What is told of a card beside its input.  Its parameters come after
        # every other: under one seed a model begins as the one that is not
        # told them, and they are drawn after it.
        if not 0.0 <= hide_identities <= 1.0:
            raise ValueError("A model hides the identities of a share of the cards: 0 to 1")
        self.reads_descriptions = bool(descriptions)
        self.hide_identities = float(hide_identities)
        if self.reads_descriptions or self.hides_identities:
            if cards is None:
                raise ValueError(
                    "A model told the cards' descriptions, or that hides their identities, "
                    "is given a card table"
                )
            self.register_buffer(CODES, cards.codes.clone())
        if self.reads_descriptions:
            assert cards is not None
            if cards.rows is None or cards.descriptions is None:
                raise ValueError("A model told the cards' descriptions is given a table of them")
            self.register_buffer(ROWS, cards.rows.clone())
            self.register_buffer(TABLE, cards.descriptions.clone())
            # A card without a description, or not in the table, adds nothing.
            self.description = nn.Linear(cards.width, embedding_dim, bias=False)
        if self.hides_identities:
            assert cards is not None
            count = len(cards.codes)
            # The cards of the table trained on: shown, with their identity,
            # in the games and the training of an update.
            self.register_buffer("card_trained", torch.zeros(count, dtype=torch.bool))
            # What is hidden now: the cards of the table so marked, and with
            # `hides_unknown` every code not in it but 0, a card not seen.
            self.register_buffer(
                "card_hidden", torch.zeros(count, dtype=torch.bool), persistent=False
            )
            self.register_buffer(
                "hides_unknown", torch.zeros((), dtype=torch.bool), persistent=False
            )
            # What a hidden card's identity is read as.
            self.hidden_identity = nn.Parameter(torch.zeros(embedding_dim))

    @property
    def hides_identities(self) -> bool:
        return self.hide_identities > 0.0

    def hide(self, hidden: Tensor) -> None:
        """Hide the identities of the cards of the table ``hidden`` marks and
        show every other, as in the games and the training of an update."""
        self.card_hidden.copy_(hidden)
        self.hides_unknown.fill_(False)

    def play(self) -> None:
        """Hide what is hidden at play: the identity of every card of the
        table not trained on, and of every code not in the table but 0."""
        if self.hides_identities:
            self.card_hidden.copy_(~self.card_trained)
            self.hides_unknown.fill_(True)

    def trained_on(self, cards: Tensor) -> None:
        """Keep the cards of the table ``cards`` marks among those trained on."""
        self.card_trained.logical_or_(cards.to(self.card_trained.device))

    def _descriptions(self) -> Tensor | None:
        """Every description of the table as the model adds it to an
        identity, worked out once a call; none where it is not told them."""
        if not self.reads_descriptions:
            return None
        return self.description(self.card_descriptions)

    def _identity(self, card: Tensor, described: Tensor | None = None) -> Tensor:
        """A card's identity as the model reads it: learnt from its code,
        hidden where the model hides it, its description added where the
        model is told them (``described``, from ``_descriptions``)."""
        identity = self.card_identity(card)
        if not (self.reads_descriptions or self.hides_identities):
            return identity
        codes = self.card_codes
        place = torch.searchsorted(codes, card).clamp_max(codes.shape[0] - 1)
        known = codes[place] == card
        if self.hides_identities:
            hidden = torch.where(known, self.card_hidden[place], self.hides_unknown) & (card != 0)
            identity = torch.where(
                hidden.unsqueeze(-1), self.hidden_identity.to(identity.dtype), identity
            )
        if self.reads_descriptions:
            if described is None:
                described = self._descriptions()
            assert described is not None
            identity = (
                identity + described[torch.where(known, self.card_description_rows[place], 0)]
            )
        return identity

    @staticmethod
    def _masked_mean(values: Tensor, mask: Tensor) -> Tensor:
        weights = mask.unsqueeze(-1).to(values.dtype)
        return (values * weights).sum(-2) / weights.sum(-2).clamp_min(1)

    @staticmethod
    def _present(lengths: Tensor, width: int) -> Tensor:
        return torch.arange(width, device=lengths.device).unsqueeze(0) < lengths.unsqueeze(1)

    def _card_representation(
        self, card: Tensor, categories: Tensor, numeric: Tensor, described: Tensor | None = None
    ) -> Tensor:
        """Combine the identity index with structured, public card metadata."""
        if categories.shape[-1] != CATEGORY_COUNT or numeric.shape[-1] != NUMERIC_COUNT:
            raise ValueError("Unexpected card metadata schema")
        setcodes = categories[..., 4:8]
        set_mask = (setcodes != 0).unsqueeze(-1)
        set_embedding = self.setcode_embedding(setcodes)
        set_embedding = (set_embedding * set_mask).sum(-2) / set_mask.sum(-2).clamp_min(1)
        return torch.cat(
            (
                self._identity(card, described),
                self.card_kind_embedding(categories[..., 0]),
                self.attribute_embedding(categories[..., 1]),
                self.race_embedding(categories[..., 2]),
                self.level_embedding(categories[..., 3]),
                set_embedding,
                self.card_numeric(numeric),
            ),
            dim=-1,
        )

    def _placed_cards(
        self, tensor: dict[str, Tensor], prefix: str, described: Tensor | None = None
    ) -> Tensor:
        """Encode the cards under ``prefix``, each where it lies and as it is."""
        return self.placed_card_encoder(
            torch.cat(
                (
                    self._card_representation(
                        tensor[f"{prefix}card"],
                        tensor[f"{prefix}categories"],
                        tensor[f"{prefix}numeric"],
                        described,
                    ),
                    self.location_embedding(tensor[f"{prefix}location"]),
                    self.position_embedding(tensor[f"{prefix}position"]),
                    self.controller_embedding(tensor[f"{prefix}controller"]),
                    tensor[f"{prefix}status"],
                ),
                dim=-1,
            )
        )

    def _history(self, tensor: dict[str, Tensor], described: Tensor | None = None) -> Tensor:
        """The seat's own earlier answers, oldest first."""
        lengths = tensor["history_lengths"]
        steps = torch.cat(
            (
                self._identity(tensor["history_card"], described),
                self.location_embedding(tensor["history_location"]),
                self.action_kind_embedding(tensor["history_kind"]),
                tensor["history_scalars"],
            ),
            dim=-1,
        )
        # The batch is given packed, every row at the length the batch is
        # padded to: the same steps of the same rows.  On a card cuDNN then
        # works them out its standard way.  For a padded batch of an LSTM of
        # at most 192 hidden units PyTorch has it use a kernel that takes ten
        # times as long at the batches of an update (docs/commands.md, "The
        # LSTM is given a packed batch").  A row's history has a step at
        # least, an empty one where it has no earlier answer.
        whole = torch.full((steps.shape[0],), steps.shape[1], dtype=torch.int64)
        packed, _ = self.history(pack_padded_sequence(steps, whole, batch_first=True))
        output, _ = pad_packed_sequence(packed, batch_first=True)
        batch = torch.arange(output.shape[0], device=output.device)
        return output[batch, lengths.clamp_min(1) - 1] * (lengths > 0).unsqueeze(-1)

    def encode_state(
        self, tensor: dict[str, Tensor], described: Tensor | None = None
    ) -> tuple[Tensor, Tensor, Tensor]:
        """The state, the token of each of its cards, and which are on the field."""
        if described is None:
            described = self._descriptions()
        board_tokens = self._placed_cards(tensor, "state_", described)
        present = self._present(tensor["state_lengths"], board_tokens.shape[1])
        context = self.context_encoder(self._placed_cards(tensor, "context_", described))
        context = context * tensor["context_mask"].unsqueeze(-1)

        def read(part: str, width: int, reading: Callable[[], Tensor]) -> Tensor:
            if part in self.leave_out:
                return board_tokens.new_zeros(board_tokens.shape[0], width)
            return reading()

        state = self.state(
            torch.cat(
                (
                    self._masked_mean(board_tokens, present),
                    self.phase_embedding(tensor["phase"]),
                    self.decision_kind_embedding(tensor["decision_kind"]),
                    self.hint_embedding(tensor["decision_hint"]),
                    tensor["scalars"],
                    context.flatten(1),
                    read("history", self.hidden_dim, lambda: self._history(tensor, described)),
                ),
                dim=-1,
            )
        )
        location = tensor["state_location"]
        on_field = present & ((location == MONSTER_ZONE) | (location == SPELL_TRAP_ZONE))
        return state, board_tokens, on_field

    def _attend_board(self, action: Tensor, board_tokens: Tensor, board_mask: Tensor) -> Tensor:
        query = self.board_query(action)
        keys = self.board_key(board_tokens)
        values = self.board_value(board_tokens)
        scores = torch.einsum("bah,bch->bac", query, keys) / query.shape[-1] ** 0.5
        weights = torch.softmax(scores.masked_fill(~board_mask.unsqueeze(1), -1e9), dim=-1)
        weights = weights * board_mask.unsqueeze(1)
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-12)
        context = torch.einsum("bac,bch->bah", weights, values)
        return action + self.board_fusion(context)

    def forward(self, tensor: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        """Logits over the answers, masked where there is none, and the state's value."""
        logits, value, _ = self.evaluate(tensor)
        return logits, value

    def evaluate(
        self,
        tensor: dict[str, Tensor],
        *,
        detach_lead: bool = False,
        detach_value: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """What `forward` gives, and the lead predicted for every answer.

        An answer's lead is how much more often it wins than the pilot's
        answer to the same decision, in units of win rate, as a search
        measures it over worlds both are played out in.  0 where there is no
        answer.  With ``detach_lead`` the head reads the policy's view of an
        answer and no gradient of its own reaches that view; with
        ``detach_value`` the value head reads the state the same way.
        """
        logits, value, leads, _, _, _ = self._evaluated(
            tensor, detach_lead=detach_lead, detach_value=detach_value
        )
        return logits, value, leads

    def parts(self, tensor: dict[str, Tensor]) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """What `forward` gives, worked out the same way, and what it is
        worked out from: the state, the token of each of its cards, and each
        answer.  A head beside the network reads them
        (`ygo_training/card_head.py`)."""
        logits, value, _, state, board_tokens, action = self._evaluated(tensor)
        return logits, value, state, board_tokens, action

    def _evaluated(
        self,
        tensor: dict[str, Tensor],
        *,
        detach_lead: bool = False,
        detach_value: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        described = self._descriptions()
        state, board_tokens, board_mask = self.encode_state(tensor, described)
        scalars = tensor["action_scalars"]
        member_value = tensor["action_member_value"]
        members = self.action_member_encoder(
            torch.cat(
                (
                    self._placed_cards(tensor, "action_member_", described),
                    ((member_value & 0xFFFF).to(state.dtype) / 65535.0).unsqueeze(-1),
                    (((member_value >> 16) & 0xFFFF).to(state.dtype) / 65535.0).unsqueeze(-1),
                    tensor["action_member_order"].unsqueeze(-1),
                ),
                dim=-1,
            )
        )
        member_mask = tensor["action_member_mask"]
        action = self.action(
            torch.cat(
                (
                    # An answer without a card has none to encode.
                    self._placed_cards(tensor, "action_", described) * scalars[..., :1],
                    self.action_kind_embedding(tensor["action_kind"]),
                    scalars,
                    self._masked_mean(members, member_mask),
                    member_mask.sum(-1, keepdim=True).to(state.dtype) / 64.0,
                ),
                dim=-1,
            )
        )
        if self.board_representation == "individual-tokens":
            action = self._attend_board(action, board_tokens, board_mask)
        state_action = torch.cat(
            (state.unsqueeze(1).expand(-1, action.shape[1], -1), action), dim=-1
        )
        logits = (
            self.score(state_action).squeeze(-1).masked_fill(~tensor["action_mask"], float("-inf"))
        )
        read = state_action.detach() if detach_lead else state_action
        leads = self.lead(read).squeeze(-1).masked_fill(~tensor["action_mask"], 0.0)
        value = self.value(state.detach() if detach_value else state).squeeze(-1)
        return logits, value, leads, state, board_tokens, action
